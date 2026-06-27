"""LaMa inpainting model in MLX for Apple Silicon GPU acceleration."""

import os
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def reflect_pad(x, pad_h, pad_w):
    """Reflect-pad spatial dims of NHWC tensor."""
    if pad_h > 0:
        top = x[:, 1:pad_h + 1, :, :][:, ::-1, :, :]
        bottom = x[:, -pad_h - 1:-1, :, :][:, ::-1, :, :]
        x = mx.concatenate([top, x, bottom], axis=1)
    if pad_w > 0:
        left = x[:, :, 1:pad_w + 1, :][:, :, ::-1, :]
        right = x[:, :, -pad_w - 1:-1, :][:, :, ::-1, :]
        x = mx.concatenate([left, x, right], axis=2)
    return x


class ReflectConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, bias=True):
        super().__init__()
        self.pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, bias=bias)

    def __call__(self, x):
        x = reflect_pad(x, self.pad, self.pad)
        return self.conv(x)


class SpectralTransform(nn.Module):
    """FFT-based spectral convolution."""

    def __init__(self, half_ch):
        super().__init__()
        self.conv = nn.Conv2d(half_ch * 2, half_ch * 2, 1, bias=True)
        self.half_ch = half_ch

    def __call__(self, x):
        B, H, W, C = x.shape
        # NHWC -> NCHW for FFT (operates on last 2 dims)
        x = mx.transpose(x, (0, 3, 1, 2))
        x_ft = mx.fft.rfft2(x)
        x_real = mx.real(x_ft)
        x_imag = mx.imag(x_ft)
        # Concat real+imag along channels: (B, 2C, H, W//2+1)
        x_cat = mx.concatenate([x_real, x_imag], axis=1)
        # NCHW -> NHWC for conv
        x_cat = mx.transpose(x_cat, (0, 2, 3, 1))
        x_cat = nn.relu(self.conv(x_cat))
        # NHWC -> NCHW
        x_cat = mx.transpose(x_cat, (0, 3, 1, 2))
        # Split back
        x_real = x_cat[:, :C]
        x_imag = x_cat[:, C:]
        x_complex = x_real + 1j * x_imag
        x = mx.fft.irfft2(x_complex, s=(H, W))
        # NCHW -> NHWC
        return mx.transpose(x, (0, 2, 3, 1))


class ConvG2G(nn.Module):
    """Global-to-global conv with spectral transform."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        half = in_ch // 2
        self.conv1 = nn.Conv2d(in_ch, half, 1, bias=True)
        self.fu = SpectralTransform(half)
        self.conv2 = nn.Conv2d(half, out_ch, 1, bias=False)

    def __call__(self, x):
        y = nn.relu(self.conv1(x))
        y = y + self.fu(y)
        return self.conv2(y)


class FFC(nn.Module):
    """Fast Fourier Convolution with local and global branches."""

    def __init__(self, in_l, in_g, out_l, out_g, stride=1, bias=True):
        super().__init__()
        self.convl2l = ReflectConv2d(in_l, out_l, 3, stride=stride, bias=bias)
        self.convl2g = ReflectConv2d(in_l, out_g, 3, stride=stride, bias=bias)
        self.convg2l = ReflectConv2d(in_g, out_l, 3, stride=stride, bias=bias)
        self.convg2g = ConvG2G(in_g, out_g)

    def __call__(self, x_l, x_g):
        out_l = self.convl2l(x_l) + self.convg2l(x_g)
        out_g = self.convl2g(x_l) + self.convg2g(x_g)
        return out_l, out_g


class FFCBlock(nn.Module):
    """FFC + BatchNorm + ReLU."""

    def __init__(self, ch_l, ch_g):
        super().__init__()
        self.ffc = FFC(ch_l, ch_g, ch_l, ch_g, bias=False)
        self.bn_l = nn.BatchNorm(ch_l)
        self.bn_g = nn.BatchNorm(ch_g)

    def __call__(self, x_l, x_g):
        x_l, x_g = self.ffc(x_l, x_g)
        x_l = nn.relu(self.bn_l(x_l))
        x_g = nn.relu(self.bn_g(x_g))
        return x_l, x_g


class FFCResBlock(nn.Module):
    """FFC ResNet block with residual connection."""

    def __init__(self, ch_l=128, ch_g=384):
        super().__init__()
        self.conv1 = FFCBlock(ch_l, ch_g)
        self.conv2 = FFCBlock(ch_l, ch_g)

    def __call__(self, x_l, x_g):
        res_l, res_g = x_l, x_g
        x_l, x_g = self.conv1(x_l, x_g)
        x_l, x_g = self.conv2(x_l, x_g)
        return x_l + res_l, x_g + res_g


class UpsampleBlock(nn.Module):
    """ConvTranspose + BN + ReLU with output_padding=1."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, bias=True)
        self.bn = nn.BatchNorm(out_ch)

    def __call__(self, x):
        x = self.conv_t(x)
        # output_padding=1: add one row and one column
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return nn.relu(self.bn(x))


class LaMa(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc0 = ReflectConv2d(4, 64, 7, stride=1, bias=True)
        self.enc1 = ReflectConv2d(64, 128, 3, stride=2, bias=True)
        self.enc2 = ReflectConv2d(128, 256, 3, stride=2, bias=True)
        # Transition to FFC branches
        self.trans_l = ReflectConv2d(256, 128, 3, stride=2, bias=True)
        self.trans_g = ReflectConv2d(256, 384, 3, stride=2, bias=True)
        # 18 FFC ResNet blocks
        self.ffc_blocks = [FFCResBlock(128, 384) for _ in range(18)]
        # Decoder
        self.up0 = UpsampleBlock(512, 256)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        # Final
        self.final_conv = ReflectConv2d(64, 3, 7, stride=1, bias=True)

    def __call__(self, image, mask):
        """
        image: (B, H, W, 3) float32 [0,1]
        mask:  (B, H, W, 1) float32 [0,1]
        returns: (B, H, W, 3) float32 [0,1]
        """
        masked = image * (1.0 - mask)
        x = mx.concatenate([masked, mask], axis=-1)

        # Encoder
        x = nn.relu(self.enc0(x))
        x = nn.relu(self.enc1(x))
        x = nn.relu(self.enc2(x))

        # Transition
        x_l = nn.relu(self.trans_l(x))
        x_g = nn.relu(self.trans_g(x))

        # FFC ResNet blocks
        for block in self.ffc_blocks:
            x_l, x_g = block(x_l, x_g)

        # Concat local + global
        x = mx.concatenate([x_l, x_g], axis=-1)

        # Decoder
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)

        # Final
        x = mx.sigmoid(self.final_conv(x))
        return x


def _onnx_weight_name(onnx_model, node_output_pattern, op_type='Conv'):
    """Find ONNX initializer name for a given node output pattern."""
    for n in onnx_model.graph.node:
        if n.op_type == op_type and re.search(node_output_pattern, n.output[0]):
            return n.input[1], (n.input[2] if len(n.input) > 2 else None)
    return None, None


def load_from_onnx(onnx_path):
    """Load LaMa model with weights from ONNX file."""
    import onnx
    from onnx import numpy_helper

    onnx_model = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in onnx_model.graph.initializer}

    # Build node output -> weight mapping
    wmap = {}
    for n in onnx_model.graph.node:
        out = n.output[0] if n.output else ""
        if n.op_type in ('Conv', 'ConvTranspose') and 'generator/model' in out:
            m = re.search(r'/generator/model/(.*?)/(Conv|ConvTranspose)_output', out)
            if m:
                path = m.group(1)
                wmap[path + '.w'] = n.input[1]
                if len(n.input) > 2:
                    wmap[path + '.b'] = n.input[2]
        elif n.op_type == 'BatchNormalization' and 'generator/model' in out:
            m = re.search(r'/generator/model/(.*?)/BatchNormalization_output', out)
            if m:
                path = m.group(1)
                wmap[path + '.weight'] = n.input[1]
                wmap[path + '.bias'] = n.input[2]
                wmap[path + '.running_mean'] = n.input[3]
                wmap[path + '.running_var'] = n.input[4]

    def get_w(path):
        return inits[wmap[path]]

    def conv_w(path):
        """ONNX Conv weight OIHW -> MLX OHWI."""
        return np.transpose(get_w(path + '.w'), (0, 2, 3, 1))

    def conv_t_w(path):
        """ONNX ConvTranspose weight IOHW -> MLX OHWI."""
        return np.transpose(get_w(path + '.w'), (1, 2, 3, 0))

    def conv_b(path):
        return get_w(path + '.b')

    def bn_params(path):
        return (get_w(path + '.weight'), get_w(path + '.bias'),
                get_w(path + '.running_mean'), get_w(path + '.running_var'))

    model = LaMa()

    # Helper to set conv weights
    def set_conv(module, path, transpose=False):
        w_fn = conv_t_w if transpose else conv_w
        module.conv.weight = mx.array(w_fn(path))
        if path + '.b' in wmap:
            module.conv.bias = mx.array(conv_b(path))

    def set_conv_direct(module, path, transpose=False):
        w_fn = conv_t_w if transpose else conv_w
        module.weight = mx.array(w_fn(path))
        if path + '.b' in wmap:
            module.bias = mx.array(conv_b(path))

    def set_bn(module, path):
        w, b, rm, rv = bn_params(path)
        module.weight = mx.array(w)
        module.bias = mx.array(b)
        module.running_mean = mx.array(rm)
        module.running_var = mx.array(rv)

    # Encoder
    set_conv(model.enc0, 'model.1/ffc/convl2l')
    set_conv(model.enc1, 'model.2/ffc/convl2l')
    set_conv(model.enc2, 'model.3/ffc/convl2l')

    # Transition
    set_conv(model.trans_l, 'model.4/ffc/convl2l')
    set_conv(model.trans_g, 'model.4/ffc/convl2g')

    # FFC blocks (model.5 to model.22)
    for i, block in enumerate(model.ffc_blocks):
        mi = i + 5
        for cv_name, cv_block in [('conv1', block.conv1), ('conv2', block.conv2)]:
            prefix = f'model.{mi}/{cv_name}'
            # FFC convolutions
            set_conv(cv_block.ffc.convl2l, f'{prefix}/ffc/convl2l')
            set_conv(cv_block.ffc.convl2g, f'{prefix}/ffc/convl2g')
            set_conv(cv_block.ffc.convg2l, f'{prefix}/ffc/convg2l')
            # ConvG2G
            set_conv_direct(cv_block.ffc.convg2g.conv1, f'{prefix}/ffc/convg2g/conv1/conv1.0')
            set_conv_direct(cv_block.ffc.convg2g.fu.conv, f'{prefix}/ffc/convg2g/fu/conv_layer')
            set_conv_direct(cv_block.ffc.convg2g.conv2, f'{prefix}/ffc/convg2g/conv2')
            # BatchNorm
            set_bn(cv_block.bn_l, f'{prefix}/bn_l')
            set_bn(cv_block.bn_g, f'{prefix}/bn_g')

    # Decoder
    set_conv_direct(model.up0.conv_t, 'model.24', transpose=True)
    set_bn(model.up0.bn, 'model.25')
    set_conv_direct(model.up1.conv_t, 'model.27', transpose=True)
    set_bn(model.up1.bn, 'model.28')
    set_conv_direct(model.up2.conv_t, 'model.30', transpose=True)
    set_bn(model.up2.bn, 'model.31')

    # Final conv
    set_conv(model.final_conv, 'model.34')

    model.eval()
    return model


def save_mlx_weights(model, path):
    """Save model weights as npz for fast loading."""
    from mlx.utils import tree_flatten
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        weights[k] = np.array(v)
    np.savez(path, **weights)


def load_mlx_weights(path):
    """Load model from pre-converted npz weights."""
    model = LaMa()
    data = np.load(path)
    weights = [(k, mx.array(data[k])) for k in data.files]
    model.load_weights(weights)
    model.eval()
    return model


def convert_and_save(onnx_path, output_path):
    """One-time conversion from ONNX to MLX weights."""
    print(f"Loading ONNX model from {onnx_path}...")
    model = load_from_onnx(onnx_path)
    print(f"Saving MLX weights to {output_path}...")
    save_mlx_weights(model, output_path)
    print("Done.")
    return model


def predict(model, image_np, mask_np):
    """
    Run LaMa inpainting.
    image_np: (H, W, 3) float32 [0,1]
    mask_np: (H, W, 1) float32 binary
    Returns: (H, W, 3) float32 [0,1]
    """
    image = mx.array(image_np[np.newaxis])
    mask = mx.array(mask_np[np.newaxis])
    result = model(image, mask)
    mx.eval(result)
    return np.array(result[0])
