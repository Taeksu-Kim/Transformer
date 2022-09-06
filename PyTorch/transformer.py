import math
import torch
import torch.nn as nn

def PositionalEncoding(max_seq_len, d_model):
    '''
    PE_(pos, 2i)   =  sin(pos / power(10000, 2i / d_model))
    PE_(pos, 2i+1) =  cos(pos / power(10000, 2i / d_model))
    '''
    pe = torch.zeros([max_seq_len, d_model])
    position = torch.arange(max_seq_len).unsqueeze(1).repeat(1, d_model) # pos, [seq_len, d_model]
    div_value = torch.pow(10000, torch.arange(0, d_model, 2) / d_model) # power(10000, 2i / d_model)
    pe[:, 0::2] = torch.sin(position[:, 0::2] / div_value) # sin for 2i
    pe[:, 1::2] = torch.cos(position[:, 1::2] / div_value) # cos for 2i+1
    pe = pe.unsqueeze(0) # [bs(1), seq_len, d_model]
    
    return pe

def get_attn_pad_mask(key_inputs, pad_id, query_len):
    return key_inputs.eq(pad_id).unsqueeze(1).expand(-1, query_len, -1) # [bs, query_len, key_len]

# decoder mask for the back of current positions. shape of attention matrix
# it's used for decoder self attention.
def get_subsequent_mask(inputs):
    subsequent_mask = torch.ones_like(inputs).unsqueeze(-1).expand(inputs.size(0), inputs.size(1), inputs.size(1)) # [bs, query_len, key_len]
    subsequent_mask = subsequent_mask.triu(diagonal=1) # like 0, 1, 1, 1...  subsequent_mask
    return subsequent_mask                             #      0, 0, 1, 1... 

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.d_model = config.d_model
        self.num_att_heads = config.num_att_heads
        assert self.d_model % self.num_att_heads == 0, "d_model({}) % num_att_heads({}) = {}. It should be 0.".format(self.d_model, self.num_att_heads, self.d_model % self.num_att_heads)
        
        self.d_head = int(self.d_model / self.num_att_heads)
        
        self.query_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.key_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.value_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)

        self.scaled_dot_attn = ScaledDotProductAttention(config, self.d_head)
        self.linear = nn.Linear(self.d_head * self.num_att_heads, self.d_model)

    def forward(self, query, key, value, attn_mask):
        batch_size = query.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, query_len, d_head]
        key = self.key_proj(key).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, key_len, d_head]
        value = self.value_proj(value).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, value_len, d_head]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_att_heads, 1, 1) # [bs, query_len, key_len] -> [bs, num_heads, query_len, key_len]

        context, attn_prob = self.scaled_dot_attn(query, key, value, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_att_heads * self.d_head)
        
        output = self.linear(context)
        
        return output, attn_prob

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config, d_head):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config
        self.scale = d_head ** 0.5

    def forward(self, query, key, value, attn_mask):

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale # [bs, num_heads, query_len, key_len]
        
        scores.masked_fill_(attn_mask, -1e9)
        
        attn_prob = nn.Softmax(dim=-1)(scores)
        # 성능 관련 실험 필요. 허깅페이스에서는 dropout 사용함
        # attn_prob = nn.Dropout(self.config.drop_out_raito)(attn_prob)
        context = torch.matmul(attn_prob, value) # [bs, num_heads, query_len, d_head]
                                                  
        return context, attn_prob

class PoswiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForward, self).__init__()      

        self.feed_forward = nn.Sequential(nn.Linear(config.d_model, config.feed_forward_dim),
                                          nn.Dropout(config.drop_out_raito),
                                          nn.ReLU(),
                                          nn.Linear(config.feed_forward_dim, config.d_model),
                                          nn.Dropout(config.drop_out_raito))

    def forward(self, inputs):
        return self.feed_forward(inputs)

class AddNorm(nn.Module):
    def __init__(self, layer, d_model):
        super(AddNorm, self).__init__()
        self.layer = layer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, *args):
        residual = args[0]
        output = self.layer(*args)

        if isinstance(output, tuple):
            return self.layer_norm(output[0] + residual), output[1]
        else:
            return self.layer_norm(output + residual)

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.sqrt_dim = math.sqrt(config.d_model)
        self.pos_encoding = PositionalEncoding(config.max_seq_len, config.d_model)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_enc_layers)]
        )

    def forward(self, enc_inputs, self_attn_mask):
        outputs = self.word_embedding(enc_inputs) * self.sqrt_dim + self.pos_encoding.to(enc_inputs.device)
        
        self_attn_probs = []
        for layer in self.layers:
            outputs, self_attn_prob = layer(outputs, self_attn_mask)
            self_attn_probs.append(self_attn_prob)
        
        return outputs, self_attn_probs

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(config), config.d_model)
        self.feed_forward = AddNorm(PoswiseFeedForward(config), config.d_model)

    def forward(self, inputs, self_attn_mask):
        outputs, self_attn_prob = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs = self.feed_forward(outputs)
        return outputs, self_attn_prob

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.sqrt_dim = math.sqrt(config.d_model)
        self.pos_encoding = PositionalEncoding(config.max_dec_len, config.d_model)

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_dec_layers)]
        )

        self.fc = nn.Linear(config.d_model, config.vocab_size)

    def decoder_step(self,
                     dec_inputs,
                     enc_outputs,
                     enc_inputs):

        dec_outputs = self.word_embedding(dec_inputs) * self.sqrt_dim + self.pos_encoding[:, :dec_inputs.size(1)].to(dec_inputs.device)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, self.config.pad_id, dec_inputs.size(1))
        dec_self_attn_subsequent_mask = get_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_cross_attn_mask = get_attn_pad_mask(enc_inputs, self.config.pad_id, dec_inputs.size(1))

        self_attn_probs, cross_attn_probs = [], []
        for layer in self.layers:
            dec_outputs, self_attn_prob, cross_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_cross_attn_mask)
            self_attn_probs.append(self_attn_prob)
            cross_attn_probs.append(cross_attn_prob)

        return dec_outputs, self_attn_probs, cross_attn_probs

    def forward(self,
                dec_inputs,
                enc_outputs,
                enc_inputs):
       

        if dec_inputs is not None:
            dec_outputs, self_attn_probs, cross_attn_probs = self.decoder_step(dec_inputs=dec_inputs,
                                                                               enc_outputs=enc_outputs,
                                                                               enc_inputs=enc_inputs)
            dec_outputs = self.fc(dec_outputs)

        else:
            dec_inputs = torch.zeros([enc_outputs.size(0), self.config.max_dec_len], device=enc_outputs.device).long()
            dec_inputs = dec_inputs.fill_(self.config.pad_id)
            dec_inputs[:, 0] = self.config.bos_id

            dec_outputs = []
            for dec_idx in range(1, self.config.max_dec_len):
                dec_output, self_attn_probs, cross_attn_probs = self.decoder_step(dec_inputs=dec_inputs[:, :dec_idx],
                                                                                  enc_outputs=enc_outputs,
                                                                                  enc_inputs=enc_inputs)
                dec_output = self.fc(dec_output)                
                dec_outputs.append(dec_output[:, -1, :])                
                dec_inputs[:, dec_idx] = dec_outputs[-1].argmax(dim=-1)
                
            dec_outputs = torch.stack(dec_outputs, dim=1)


        return dec_outputs, self_attn_probs, cross_attn_probs

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attention = AddNorm(MultiHeadAttention(config), config.d_model)
        self.cross_attention = AddNorm(MultiHeadAttention(config), config.d_model)
        
        self.feed_forward = AddNorm(PoswiseFeedForward(config), config.d_model)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, cross_attn_mask):
        
        outputs, self_attn_prob = self.self_attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        outputs, cross_attn_prob = self.cross_attention(outputs, enc_outputs, enc_outputs, cross_attn_mask)
        outputs = self.feed_forward(outputs)
        
        return outputs, self_attn_prob, cross_attn_prob

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        if config.use_decoder == True:
            self.decoder = TransformerDecoder(config)
            self.fc = nn.Linear(config.d_model, config.vocab_size)
    def forward(self, 
                enc_inputs, 
                dec_inputs=None, 
                enc_self_attn_mask=None,
                dec_self_attn_mask=None,
                dec_cross_attn_mask=None):
        
        if enc_self_attn_mask == None:
            enc_self_attn_mask = get_attn_pad_mask(enc_inputs, self.config.pad_id, enc_inputs.size(1))

        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs, enc_self_attn_mask)
        
        if self.config.use_decoder == False:
            return {'encoder_hidden_states' : enc_outputs, 
                    'encoder_self_attention_prob' : enc_self_attn_probs}
        
        dec_outputs, dec_self_attn_probs, dec_cross_attn_probs = self.decoder(dec_inputs,
                                                                              enc_outputs, 
                                                                              enc_inputs)
        
        return {'decoder_hidden_states' : dec_outputs, 
                'encoder_self_attention_prob' : enc_self_attn_probs, 
                'decoder_self_attention_prob' : dec_self_attn_probs, 
                'decoder_cross_attention_prob' : dec_cross_attn_probs}