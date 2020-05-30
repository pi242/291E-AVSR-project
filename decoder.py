import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CrossModalTDecoderLayer(nn.Module):
    """
    No self attention, only for use with CTC decoding
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossModalTDecoderLayer, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.nhead = nhead
        self.oa_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ov_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.oav_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.norm_a = nn.LayerNorm(d_model)
        self.dropout_a = nn.Dropout(p=dropout)
        self.norm_v = nn.LayerNorm(d_model)
        self.dropout_v = nn.Dropout(p=dropout)
        self.norm_av = nn.LayerNorm(d_model)
        self.dropout_av = nn.Dropout(p=dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu

    def create_padding_mask(self, kdims, max_len):
        """
        kdims = batch_size x 1 tensor
        max_len = 512
        """
        mask = torch.zeros((kdims.size(0), max_len), dtype=torch.uint8)
        for i in range(kdims.size(0)):
            mask[i, kdims[i]:] = 1
        return mask

    def create_attn_mask(self, qdims, kdims, qmax_len, kmax_len):
        mask = torch.zeros((qdims.size(0) * self.nhead, qmax_len, kmax_len))
        for i in range(qdims.size(0)):
            mask[(i * self.nhead):(i * self.nhead + self.nhead), qdims[i]:, :] = float('-inf')
        return mask


    # def forward(self, oa, ov, audio_dims, video_dims):
    def forward(self, oa, ov):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        # op_oa1 = self.oa_attn(oa, oa, oa, key_padding_mask=self.create_padding_mask(audio_dims, oa.size(0)))[0]
        op_oa1 = self.oa_attn(oa, oa, oa)[0]
        op_oa = oa + self.dropout_a(op_oa1)
        op_oa = self.norm_a(op_oa)

        # op_ov1 = self.ov_attn(ov, ov, ov, key_padding_mask=self.create_padding_mask(video_dims, ov.size(0)))[0]
        op_ov1 = self.ov_attn(ov, ov, ov)[0]
        op_ov = ov + self.dropout_v(op_ov1)
        op_ov = self.norm_v(op_ov)

        # op_oav1 = self.oav_attn(oa, ov, ov, key_padding_mask=self.create_padding_mask(video_dims, ov.size(0)), attn_mask=self.create_attn_mask(audio_dims, video_dims, oa.size(0), ov.size(0)))[0]
        op_oav1 = self.oav_attn(oa, ov, ov)[0]
        op_oav = oa + self.dropout_av(op_oav1)
        op_oav = self.norm_av(op_oav)

        combined = torch.cat((op_oa, op_ov, op_oav), dim=0)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(combined))))
        tgt = combined + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class TDecoder_CM(nn.Module):

    def __init__(self, emb_size, ntokens, nhead, nhid, nlayers):
        """
        emb_size
        nhead: Number of transformer heads in the encoder
        nhid: Number of hidden units in transformer encoder layer
        ntokens: vocab size of audio ???
        nlayer: Number of layers in transformer encoder
        """
        super(TDecoder_CM, self).__init__()
        from torch.nn import MultiheadAttention
        otherlayer = nn.TransformerDecoderLayer(emb_size, nhead, nhid)
        self.cmlayer = CrossModalTDecoderLayer(emb_size, nhead, nhid)
        self.otherlayers = nn.TransformerDecoder(otherlayer, nlayers)

        self.linear_op = nn.Linear(emb_size, ntokens + 1)  # + 1 for CTC !!!!
        
        
    def forward(self, oa, ov):
        """
        oa: tensor of shape (204, batch_size, emb_size)
        ov: tensor of shape (155, batch_size, emb_size)
        Returns:
            output: tensor of shape (seq_len, batch_size, emb_size)
        """
        
        op1 = self.cmlayer.forward(oa, ov)
        output = self.otherlayers(op1, op1)
        output = self.linear_op(output)
        return output
    
class GeneralTDecoderLayer(nn.Module):
    """
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(GeneralTDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt, memory, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(tgt, memory, memory, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = tgt + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    
class TDecoder_Gen(nn.Module):

    def __init__(self, emb_size, ntokens, nhead, nhid, nlayers):
        """
        emb_size
        nhead: Number of transformer heads in the encoder
        nhid: Number of hidden units in transformer encoder layer
        ntokens: vocab size of audio ???
        nlayer: Number of layers in transformer encoder
        """
        super(TDecoder_Gen, self).__init__()
        decoderlayer = GeneralTDecoderLayer(emb_size, nhead, nhid)
        self.oa_layers = _get_clones(decoderlayer, nlayers)
        self.ov_layers = _get_clones(decoderlayer, nlayers)
        self.oav_layers = _get_clones(decoderlayer, nlayers)

        self.linear_op = nn.Linear(emb_size, ntokens + 1)  # + 1 for CTC !!!!
        self.nlayers = nlayers
        
    def forward(self, oa, ov):
        """
        oa: tensor of shape (204, batch_size, emb_size)
        ov: tensor of shape (155, batch_size, emb_size)
        Returns:
            output: tensor of shape (seq_len, batch_size, emb_size)
        """
        oav = oa
        for num in range(self.nlayers):
            tempa = self.oa_layers[num].forward(oa, oa)
            # print(tempa.shape)
            tempv = self.ov_layers[num].forward(ov, ov)
            # print(tempv.shape)
            tempav = self.oav_layers[num].forward(oav, ov)
            # print(tempav.shape)
            oa = tempa
            ov = tempv
            oav = tempav
        combined = torch.cat((oa, ov, oav), dim=0)
        output = self.linear_op(combined)
        return output
