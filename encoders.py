import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Adds positional embedding to the input for conditioning on time. 
    This is already implemented for you, but you can try other variants of positional encoding.
    Read the paper "Attention is all you need" for more details on this. 
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: tensor of shape (seq_len, batch_size, embedding_size)
        Returns:
            x: tensor of shape (seq_len, batch_size, embedding_size)
        """
        return self.pe[:x.size(0), :]

class AudioTEncoder(nn.Module):

    def __init__(self, ip_size, emb_size, nhead, nhid, nlayers, dropout=0.1):
        """
        ip_size: MFCC features
        emb_size: size after initial embedding using linear layer
        nhead: Number of transformer heads in the encoder
        nhid: Number of hidden units in transformer encoder layer
        nlayer: Number of layers in transformer encoder
        """
        super(AudioTEncoder, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(ip_size, emb_size)
        self.layer_norm_in = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.pe = PositionalEncoding(emb_size)

        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid)
        self.trans_encoder = TransformerEncoder(encoder_layers, nlayers)

    def create_padding_mask(self, audio_dims, max_len):
        """
        audio_dims = batch_size x 1 tensor
        max_len = 204
        """
        mask = torch.zeros((audio_dims.size(0), max_len), dtype=torch.uint8)
        for i in range(audio_dims.size(0)):
            mask[i, audio_dims[i]:] = 1
        return mask
        
    def forward(self, src, audio_dims):
        """
        src: tensor of shape (seq_len, batch_size, ip_size)
        audio_dims: (batch_size, 1) tensor
        Returns:
            output: tensor of shape (seq_len, batch_size, emb_size)
        """
        temp = self.linear_in(src)
        # print(f'temp shape = {temp.shape}')
        encoded = self.layer_norm_in(temp) # seq_len x batch_size x emb_size
        # print(f'encoded shape = {encoded.shape}')
        pos_encoded = self.dropout(encoded + self.pe(encoded)) # seq_len x batch_size x emb_size
        output = self.trans_encoder(pos_encoded, src_key_padding_mask=self.create_padding_mask(audio_dims, src.size(0))) # seq_len x batch_size x emb_size
        return output

class VideoTEncoder(nn.Module):

    def __init__(self, ip_size, emb_size, nhead, nhid, nlayers, dropout=0.1):
        """
        ip_size: MFCC features
        emb_size: size after initial embedding using linear layer
        nhead: Number of transformer heads in the encoder
        nhid: Number of hidden units in transformer encoder layer
        nlayer: Number of layers in transformer encoder
        """
        super(VideoTEncoder, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(ip_size, emb_size)
        self.layer_norm_in = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.pe = PositionalEncoding(emb_size)

        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid)
        self.trans_encoder = TransformerEncoder(encoder_layers, nlayers)

    def create_padding_mask(self, video_dims, max_len):
        """
        audio_dims = batch_size x 1 tensor
        max_len = 512
        """
        mask = torch.zeros((video_dims.size(0), max_len), dtype=torch.uint8)
        for i in range(video_dims.size(0)):
            mask[i, video_dims[i]:] = 1
        return mask
        
    def forward(self, src, video_dims):
        """
        src: tensor of shape (seq_len, batch_size, ip_size)
        video_dims: (batch_size, 1) tensor
        Returns:
            output: tensor of shape (seq_len, batch_size, emb_size)
        """
        
        temp = self.linear_in(src)
        encoded = self.layer_norm_in(temp) # seq_len x batch_size x emb_size
        pos_encoded = self.dropout(encoded + self.pe(encoded)) # seq_len x batch_size x emb_size
        output = self.trans_encoder(pos_encoded, src_key_padding_mask=self.create_padding_mask(video_dims, src.size(0))) # seq_len x batch_size x emb_size
        return output