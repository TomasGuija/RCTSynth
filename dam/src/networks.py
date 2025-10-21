import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from . import layers
from .utils import default_unet_features, default_encoder_features
from .modelio import LoadableModel, store_config_args


class Encoder(nn.Module):
    """
    Inference network based on unet architecture. Layer features can be specified directly as a list of  
    encoder features or as a single integer along with a number of unet levels. The default network
    features per layer (when no options are specified) are:

        encoder: [16, 32, 32, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 latent_dim=32,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 last_level=4):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [encoder feats], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            last_level: Number of features in the last encoder level.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_encoder_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        enc_nf = nb_features
        self.max_pool = max_pool
        self.nb_levels = int(len(enc_nf) / nb_conv_per_level)

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # the inference network takes planning & repeat CTs and outputs
        # the parameters of a Gaussian distribution in the latent space
        # last layers to predict distribution parameters q(z|x)
        self.last_conv = ConvBlock(ndims, prev_nf, last_level)
        hidden_dim = tuple(int(i/(self.max_pool**self.nb_levels)) for i in inshape)
        self.mu = nn.Linear(last_level*np.prod(hidden_dim), latent_dim)
        self.logvar = nn.Linear(last_level*np.prod(hidden_dim), latent_dim)

    def reparametrize(self, mu, logvar):

        # reparametrization trick
        stdev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stdev)
        latents = epsilon * stdev + mu
        return latents, mu, stdev

    def forward(self, x):

        # encoder forward pass
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x = self.pooling[level](x)

        # project to latent variables
        x = self.last_conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.reparametrize(self.mu(x), self.logvar(x))


class Generator(nn.Module):
    """
    Dual-encoder U-Net:
      - Image encoder -> prior p(z|x), bottleneck features
      - Anatomy encoder (masks) -> bottleneck features
      - Concatenate [image bottleneck, z, anatomy bottleneck] -> decoder
    """
    def __init__(
        self,
        inshape=None,
        img_infeats=None,          
        anat_infeats=None,
        latent_dim=32,
        nb_features=None,
        nb_levels=None,
        max_pool=2,
        feat_mult=1,
        nb_conv_per_level=1,
        last_level=4,
        half_res=False,
    ):
        super().__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3], f'ndims should be 1, 2, or 3. found: {ndims}'
        self.half_res = half_res
        self.max_pool = max_pool

        # default features
        if nb_features is None:
            nb_features = default_unet_features()

        # feature list
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level)

        if isinstance(max_pool, int):
            max_pool_list = [max_pool] * self.nb_levels
        else:
            max_pool_list = max_pool
        MaxPooling = getattr(nn, f'MaxPool{ndims}d')
        self.pooling = [MaxPooling(s) for s in max_pool_list]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool_list]

        # compute bottleneck spatial shape (assumes integer max_pool scalar; if list, adapt accordingly)
        hidden_dim = tuple(int(i / (self.max_pool ** self.nb_levels)) for i in inshape)

        # -------------------------
        # IMAGE ENCODER (separate)
        # -------------------------
        prev_nf_img = img_infeats
        encoder_nfs_img = [prev_nf_img]
        self.img_encoder = nn.ModuleList()
        for level in range(self.nb_levels):
            convs = nn.ModuleList()
            for k in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + k]
                convs.append(ConvBlock(ndims, prev_nf_img, nf))
                prev_nf_img = nf
            self.img_encoder.append(convs)
            encoder_nfs_img.append(prev_nf_img)

        # prior p(z|x) heads (image stream)
        self.last_conv_img = ConvBlock(ndims, prev_nf_img, last_level)
        self.mu     = nn.Linear(last_level * np.prod(hidden_dim), latent_dim)
        self.logvar = nn.Linear(last_level * np.prod(hidden_dim), latent_dim)

        # -------------------------
        # ANATOMY ENCODER (separate)
        # -------------------------
        prev_nf_anat = anat_infeats
        self.anat_encoder = nn.ModuleList()
        for level in range(self.nb_levels):
            convs = nn.ModuleList()
            for k in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + k]
                convs.append(ConvBlock(ndims, prev_nf_anat, nf))
                prev_nf_anat = nf
            self.anat_encoder.append(convs)
        # project anatomy bottleneck to fixed channels so it can be concatenated at bottleneck
        self.anat_bottleneck = ConvBlock(ndims, prev_nf_anat, last_level)

        # -------------------------
        # LATENT -> feature volume
        # -------------------------
        self.fcn = FCBlock(latent_dim, last_level * np.prod(hidden_dim))
        self.reshape = nn.Unflatten(1, (last_level, *hidden_dim))

        # -------------------------
        # DECODER
        # -------------------------
        # decoder starts from the image-stream bottleneck (same tensor used to compute μ/logvar)
        # we will concatenate z and (optionally) anatomy bottleneck, so grow the channel count:
        prev_nf_dec_in = prev_nf_img + last_level * 2

        encoder_nfs_img = np.flip(encoder_nfs_img)
        self.decoder = nn.ModuleList()
        prev_nf = prev_nf_dec_in
        for level in range(self.nb_levels):
            convs = nn.ModuleList()
            for k in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + k]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 1):
                prev_nf += encoder_nfs_img[level]  # image skips only

        self.remaining = nn.ModuleList()
        for nf in final_convs:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        Conv = getattr(nn, f'Conv{ndims}d')
        flow_conv = Conv(prev_nf, ndims, kernel_size=3, padding=1)
        flow_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(flow_conv.weight.shape))
        flow_conv.bias = nn.Parameter(torch.zeros_like(flow_conv.bias))
        self.flow = nn.Sequential(flow_conv, nn.Hardtanh(-2, 2))

        # cache shapes/channels for forward
        self.hidden_dim = hidden_dim
        self.last_level = last_level

    def forward(self, x_img, x_anat, latents, prior=False):
        """
        x_img:  (B, img_infeats, *inshape)
        x_anat: (B, anat_infeats, *inshape) or None
        latents: (B, latent_dim) ~ N(0, I) or any noise vector (if prior=True, reparam with μ, logvar)
        """
        # -------------------------
        # IMAGE ENCODER + skips
        # -------------------------
        xi = x_img
        x_history = [xi]
        for level, convs in enumerate(self.img_encoder):
            for conv in convs:
                xi = conv(xi)
            x_history.append(xi)
            xi = self.pooling[level](xi)

        # compute μ, logvar from image bottleneck
        h = torch.flatten(self.last_conv_img(xi), start_dim=1)
        mu, logvar = self.mu(h), self.logvar(h)

        # sample reparameterized z from prior if requested
        if prior:
            latents = latents * torch.exp(0.5 * logvar) + mu

        z = self.reshape(self.fcn(latents))  # (B, last_level, *hidden_dim)

        # -------------------------
        # ANATOMY ENCODER (bottleneck only)
        # -------------------------
        xa = x_anat
        for level, convs in enumerate(self.anat_encoder):
            for conv in convs:
                xa = conv(xa)
            xa = self.pooling[level](xa)
        a = self.anat_bottleneck(xa)  # (B, anat_bottleneck_ch, *hidden_dim)

        # -------------------------
        # Bottleneck concat: [image bottleneck, z, anatomy bottleneck]
        # -------------------------
        x = torch.cat([xi, z] + ([a] if a is not None else []), dim=1)

        # -------------------------
        # DECODER with image skips
        # -------------------------
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        for conv in self.remaining:
            x = conv(x)

        return self.flow(x), mu, logvar


class DamBase(LoadableModel):
    """
    Network for (unsupervised) generation of daily anatomical variations.
    """
    
    @store_config_args
    def __init__(self,
                 inshape,
                 latent_dim=32,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 last_level=4,
                 prior=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            src_feats: Number of planning image features. Default is 1.
            trg_feats: Number of repeat image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
            last_level: Number of features in last Unet downsampling convolution.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet models
        self.inference = Encoder(
            inshape=inshape,
            infeats=(src_feats + trg_feats),
            latent_dim=latent_dim,
            nb_features=nb_unet_features[0],
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            last_level=last_level,
        )

        self.generator = Generator(
            inshape=inshape,
            img_infeats=src_feats,
            anat_infeats=src_feats,
            nb_features=nb_unet_features,
            latent_dim=latent_dim,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            last_level=last_level,
        )

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # cache parameters
        self.bidir = bidir
        self.latent_dim = latent_dim

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.warp_organs = layers.SpatialTransformer(inshape, mode='nearest')

    def to_binary(self, cat_mask, num_organs=1):
        """
        Convert volume of categorical values to binary masks.
        """
        mask = torch.nn.functional.one_hot(cat_mask, 5).transpose(1,-1).squeeze(-1).float()
        return mask[:,1:int(1+num_organs),:,:,:]

    def encode(self, planning, repeat):
        """
        Infer q(z|x) parameters from planning and repeat images.
        """
        return self.inference(torch.cat([planning, repeat], dim=1))

    def sample(self, planning, pmask, latents, prior=True):
        """
        Sample a batch of repeat images given a set of latent variables.
        Assumes latents are sampled from N(0,1)
        """
        svf, _, _ = self.generator(
            planning, pmask.float()/4, latents, prior=prior)

        # integrate and warp image
        pos_flow, neg_flow = self.to_dvf(svf)
        return self.transformer(planning, pos_flow), pos_flow, neg_flow

    def to_dvf(self, pos_flow):
        """
        Integrate flow field to obtain forward and inverse DVFs.
        Parameters:
            svf: Pre-flow from the Unet.
        """
        # resize flow for integration
        if self.resize:
            pos_flow = self.resize(pos_flow)

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        return pos_flow, neg_flow

    def forward(self, planning, repeat, pmask, rmask):

        # encode image pair and propagate latents

        # latents, q_mu, q_logvar = self.encode(
        #    torch.cat((planning, pmask.float()/4), dim=1), torch.cat((repeat, rmask.float()/4), dim=1))

        # Only encode image information, to decouple from anatomical masks
        latents, q_mu, q_logvar = self.encode(
            planning, repeat)
        svf, p_mu, p_logvar = self.generator(planning, pmask.float()/4, latents)

        # store distribution parameters for KL loss
        param = {'q_mu':q_mu, 'q_logvar':q_logvar, 'p_mu':p_mu, 'p_logvar':p_logvar}

        # integrate flow field
        pos_flow, neg_flow = self.to_dvf(svf)

        # warp image with flow field
        rep = self.transformer(planning, pos_flow)
        inv = self.transformer(repeat, neg_flow) if self.bidir else None

        # warp organs
        mrep = self.transformer(self.to_binary(pmask), pos_flow) 
        minv = self.transformer(self.to_binary(rmask), neg_flow)  if self.bidir else None

        # return non-integrated flow field during training
        return (rep, inv, mrep, minv, svf, param) if self.bidir else (rep, mrep, svf, param)            


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm = nn.GroupNorm(4, out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.norm(self.main(x)))

class FCBlock(nn.Module):
    """
    Specific feed-forward block followed by leakyrelu for unet.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.main = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.main(x))