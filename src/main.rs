use std::cmp;
use std::env;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read, Write};
use std::path::Path;
use std::time::Instant;

use ndarray::prelude::*;
use rand::Rng;

pub struct Config {
    // transformer params
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    #[allow(dead_code)]
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

impl Config {
    pub fn from_binary<R: Read>(input: &mut BufReader<R>) -> Config {
        // load the configs from a binary
        let mut i_buffer = [0u8; std::mem::size_of::<i32>()];
        let mut header = Vec::new();
        for _ in 0..7 {
            let res = input.read_exact(&mut i_buffer);
            match res {
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
                _ => {}
            }
            res.expect("Unexpected error during read");
            let x = i32::from_le_bytes(i_buffer);
            header.push(x);
        }
        Config {
            dim: header[0],
            hidden_dim: header[1],
            n_layers: header[2],
            n_heads: header[3],
            n_kv_heads: header[4],
            vocab_size: header[5],
            seq_len: header[6],
        }
    }
}

pub struct TransformerWeights {
    // weights for the transformer model layers

    // embedding
    token_embedding_table: Array2<f32>, // (vocab_size, dim)

    // rms norms
    rms_att_weight: Array2<f32>, // (layer, dim)
    rms_ffn_weight: Array2<f32>, // (layer, dim)

    // matmuls
    wq: Array3<f32>, // (layer, dim, dim)
    wk: Array3<f32>, // (layer, dim, dim)
    wv: Array3<f32>, // (layer, dim, dim)
    wo: Array3<f32>, // (layer, dim, dim)

    // ffn
    w1: Array3<f32>, // (layer, hidden_dim, dim)
    w2: Array3<f32>, // (layer, dim, hidden_dim)
    w3: Array3<f32>, // (layer, hidden_dim, dim)

    // rms norwm
    rms_final_weight: Array1<f32>, // (dim,)

    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Array2<f32>, // (seq_len, dim/(num_heads*2))
    freq_cis_imag: Array2<f32>, // (seq_len, dim/(num_heads*2))
}

impl TransformerWeights {
    pub fn zeros(conf: &Config) -> TransformerWeights {
        // initialsize weights with all zeros
        let vocab_size = conf.vocab_size.try_into().unwrap();
        let n_layers = conf.n_layers.try_into().unwrap();
        let dim = conf.dim.try_into().unwrap();
        let hidden_dim = conf.hidden_dim.try_into().unwrap();
        let seq_len = conf.seq_len.try_into().unwrap();
        let half_dim = (conf.dim / (conf.n_heads * 2)).try_into().unwrap();

        TransformerWeights {
            token_embedding_table: Array::zeros((vocab_size, dim)),
            rms_att_weight: Array::zeros((n_layers, dim)),
            rms_ffn_weight: Array::zeros((n_layers, dim)),
            wq: Array::zeros((n_layers, dim, dim)),
            wk: Array::zeros((n_layers, dim, dim)),
            wv: Array::zeros((n_layers, dim, dim)),
            wo: Array::zeros((n_layers, dim, dim)),
            w1: Array::zeros((n_layers, hidden_dim, dim)),
            w2: Array::zeros((n_layers, dim, hidden_dim)),
            w3: Array::zeros((n_layers, hidden_dim, dim)),
            rms_final_weight: Array::zeros((dim,)),
            freq_cis_real: Array::zeros((seq_len, half_dim)),
            freq_cis_imag: Array::zeros((seq_len, half_dim)),
        }
    }

    pub fn from_binary<R: Read>(input: &mut BufReader<R>, config: &Config) -> TransformerWeights {
        // load the weights from a binary buffer reader
        let vocab_size = config.vocab_size.try_into().unwrap();
        let n_layers = config.n_layers.try_into().unwrap();
        let dim = config.dim.try_into().unwrap();
        let hidden_dim = config.hidden_dim.try_into().unwrap();
        let seq_len = config.seq_len.try_into().unwrap();
        let per_head = (config.dim / (2 * config.n_heads)).try_into().unwrap();

        let mut read_weights = |size: (usize, usize, usize)| {
            // read the number of bytes from the input buffer and return a 3d Array
            let num = size.0 * size.1 * size.2;
            let next_bytes = read_num_floats(input, num.try_into().unwrap());
            Array::from_shape_vec(size, next_bytes).unwrap()
        };

        let s2 = s![.., .., 0];
        let token_embedding_table = read_weights((vocab_size, dim, 1)).slice_move(s2);
        let rms_att_weight = read_weights((n_layers, dim, 1)).slice_move(s2);
        let wq = read_weights((n_layers, dim, dim));
        let wk = read_weights((n_layers, dim, dim));
        let wv = read_weights((n_layers, dim, dim));
        let wo = read_weights((n_layers, dim, dim));
        let rms_ffn_weight = read_weights((n_layers, dim, 1)).slice_move(s2);
        let w1 = read_weights((n_layers, hidden_dim, dim));
        let w2 = read_weights((n_layers, dim, hidden_dim));
        let w3 = read_weights((n_layers, hidden_dim, dim));
        let rms_final_weight = read_weights((dim, 1, 1)).slice_move(s![.., 0, 0]);
        let freq_cis_real = read_weights((seq_len, per_head, 1)).slice_move(s2);
        let freq_cis_imag = read_weights((seq_len, per_head, 1)).slice_move(s2);

        TransformerWeights {
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
        }
    }
}

pub struct RunState {
    x: Array1<f32>,           // activation at current t (dim,)
    xb: Array1<f32>,          // same but inside residual branch (dim,)
    xb2: Array1<f32>,         // extra buffer for convenience (dim,)
    hb: Array1<f32>,          // buffer for hidden dim in ffn (hidden_dim,)
    hb2: Array1<f32>,         // buffer for hidden dim in ffn (hidden_dim,)
    q: Array1<f32>,           // query (dim,)
    k: Array1<f32>,           // key (dim,)
    v: Array1<f32>,           // value (dim,)
    att: Array1<f32>,         // buffer for scores/attn values (seq_len,)
    logits: Array1<f32>,      // (output_logits,)
    key_cache: Array3<f32>,   // (layer, seq_len, dim)
    value_cache: Array3<f32>, // (layer, seq_len, dim)
}

impl RunState {
    fn zeros(config: &Config) -> RunState {
        let vocab_size = config.vocab_size.try_into().unwrap();
        let n_layers = config.n_layers.try_into().unwrap();
        let dim = config.dim.try_into().unwrap();
        let hidden_dim = config.hidden_dim.try_into().unwrap();
        let seq_len = config.seq_len.try_into().unwrap();
        RunState {
            x: Array::zeros((dim,)),
            xb: Array::zeros((dim,)),
            xb2: Array::zeros((dim,)),
            hb: Array::zeros((hidden_dim,)),
            hb2: Array::zeros((hidden_dim,)),
            q: Array::zeros((dim,)),
            k: Array::zeros((dim,)),
            v: Array::zeros((dim,)),
            att: Array::zeros((seq_len,)),
            logits: Array::zeros((vocab_size,)),
            key_cache: Array::zeros((n_layers, seq_len, dim)),
            value_cache: Array::zeros((n_layers, seq_len, dim)),
        }
    }
}

mod nn {
    // neural net functions

    use super::*;

    pub fn rmsnorm(x: &ArrayView1<f32>, weight: &ArrayView1<f32>) -> Array1<f32> {
        // sum of squares
        let mut ss: f32 = x.dot(x);
        let n = x.len() as f32;
        ss /= n;
        ss += 1e-5_f32;
        ss = 1. / ss.sqrt();

        // norm and scale
        let y: Array1<f32> = (x * weight).map(|a| a * ss);
        y
    }

    pub fn array_max(x: &Array1<f32>) -> f32 {
        // get the maxmimum of an array
        let m: f32 = {
            x.into_iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        };
        m
    }

    pub fn softmax(x: &Array1<f32>) -> Array1<f32> {
        // find max val for numerical stability
        let m = array_max(x);

        // exp
        let y = x.map(|a| (a - m).exp());

        // normalize
        let s = y.sum();
        let y = y.map(|a| a / s);
        y
    }

    pub fn silu(x: &f32) -> f32 {
        // x * logisitic_sigmoid(x)
        x * (1. / (1. + (-x).exp()))
    }

    pub fn transformer(token: i32, pos: i32, p: &Config, s: &mut RunState, w: &TransformerWeights) {
        let head_size = p.dim / p.n_heads;

        s.x = w
            .token_embedding_table
            .index_axis(Axis(0), token.try_into().unwrap())
            .to_owned();

        let u: usize = pos.try_into().unwrap();

        // both (dim / (head * 2)),)
        let freq_cis_real_row: ArrayView1<f32> = w.freq_cis_real.index_axis(Axis(0), u);
        let freq_cis_imag_row: ArrayView1<f32> = w.freq_cis_imag.index_axis(Axis(0), u);

        for l in 0..(p.n_layers) {
            let l_u: usize = l.try_into().unwrap();
            let att_weight = w.rms_att_weight.index_axis(Axis(0), l_u);
            s.xb = rmsnorm(&s.x.view(), &att_weight);

            // attention matmul
            s.q = s.xb.dot(&w.wq.index_axis(Axis(0), l_u).t());
            s.k = s.xb.dot(&w.wk.index_axis(Axis(0), l_u).t());
            s.v = s.xb.dot(&w.wv.index_axis(Axis(0), l_u).t());

            // apply RoPE rotation to the q and k vectors for each head

            for h in 0..(p.n_heads) {
                let h_low = h * head_size;
                let h_high = (h + 1) * head_size;
                let q = s.q.slice(s![h_low..h_high]);

                let q_prev = q.slice(s![..;2]); // (dim / (heads * 2))
                let q_next = q.slice(s![1..;2]);

                let new_q_prev = {
                    q_prev.to_owned() * freq_cis_real_row - q_next.to_owned() * freq_cis_imag_row
                };
                let new_q_next = {
                    q_prev.to_owned() * freq_cis_imag_row + q_next.to_owned() * freq_cis_real_row
                };

                s.q.slice_mut(s![h_low..h_high;2]).assign(&new_q_prev);
                s.q.slice_mut(s![(h_low + 1)..h_high;2]).assign(&new_q_next);

                let k = s.k.slice(s![h_low..h_high]);
                let k_prev = k.slice(s![..;2]);
                let k_next = k.slice(s![1..;2]);

                let new_k_prev = {
                    k_prev.to_owned() * freq_cis_real_row - k_next.to_owned() * freq_cis_imag_row
                };
                let new_k_next = {
                    k_prev.to_owned() * freq_cis_imag_row + k_next.to_owned() * freq_cis_real_row
                };

                s.k.slice_mut(s![h_low..h_high;2]).assign(&new_k_prev);
                s.k.slice_mut(s![(h_low + 1)..h_high;2]).assign(&new_k_next);
            }

            // save key and value to cache
            s.key_cache.slice_mut(s![l_u, u, ..]).assign(&s.k); // (head, seq_len, dim)
            s.value_cache.slice_mut(s![l_u, u, ..]).assign(&s.v);

            // multi-head attention
            let sqrt_head_size: f32 = (head_size as f32).sqrt();
            for h in 0..(p.n_heads) {
                let h_idx_low: usize = (h * head_size).try_into().unwrap();
                let h_idx_high: usize = ((h + 1) * head_size).try_into().unwrap();
                let q_head = s.q.slice(s![h_idx_low..h_idx_high]);
                let prev_key = s.key_cache.slice(s![l_u, ..(u + 1), h_idx_low..h_idx_high]); // (pos, head_size)

                // softmax(QK / sqrt(d))
                let attn_logits = prev_key.dot(&q_head).map(|a| (a / sqrt_head_size));
                let scores = softmax(&attn_logits);
                s.att.slice_mut(s![..(u + 1)]).assign(&scores);

                // attend over value
                let weighted = {
                    s.value_cache
                        .slice(s![l_u, ..(u + 1), h_idx_low..h_idx_high])
                        .t()
                        .dot(&s.att.slice(s![..(u + 1)]))
                }; // (head_size,)
                s.xb.slice_mut(s![h_idx_low..h_idx_high]).assign(&weighted);
            }

            // final attention matmul
            s.xb2 = w.wo.slice(s![l_u, .., ..]).dot(&s.xb);

            // residual connection
            s.x += &s.xb2;

            // ffn rmsnorm
            let rms_ffn_weight = w.rms_ffn_weight.index_axis(Axis(0), l_u);
            s.xb = rmsnorm(&s.x.view(), &rms_ffn_weight);

            // ffn
            s.hb = w.w1.slice(s![l_u, .., ..]).dot(&s.xb);
            s.hb2 = w.w3.slice(s![l_u, .., ..]).dot(&s.xb);
            s.hb = s.hb.map(nn::silu);
            s.hb *= &s.hb2;
            s.xb = w.w2.slice(s![l_u, .., ..]).dot(&s.hb);

            // residual connection
            s.x += &s.xb;
        }

        // final rms norm
        s.x = rmsnorm(&s.x.view(), &w.rms_final_weight.view());

        // logits
        s.logits = w.token_embedding_table.dot(&s.x);
    }

    pub fn argmax1(arr: &Array1<f32>) -> Result<i32, &'static str> {
        // argmax of a 1d array
        let mut cur = -f32::INFINITY;
        let mut arg: i32 = -1;
        for (idx, x) in arr.iter().enumerate() {
            if x > &cur {
                cur = *x;
                arg = idx as i32;
            }
        }
        return match arg {
            -1 => Err("No argmax found."),
            x => Ok(x),
        };
    }

    pub fn sample(ps: &Array1<f32>) -> i32 {
        // sample from a probability distribution
        let mut rng = rand::thread_rng();
        let mut sum = 0.;
        let mut ps_cum = Array1::<f32>::zeros(ps.len());
        for (idx, p) in ps.iter().enumerate() {
            sum += p;
            ps_cum[idx] = sum;
        }
        let r: f32 = rng.gen();
        let idx = ps_cum.iter().position(|&x| x > r).unwrap();
        return idx as i32;
    }
}

fn read_num_floats<R: Read>(input: &mut BufReader<R>, n: i32) -> Vec<f32> {
    // read the next n floats from a binary file
    let mut f_buffer = [0u8; std::mem::size_of::<f32>()];
    let mut results = Vec::new();
    for _ in 0..n {
        let res = input.read_exact(&mut f_buffer);
        match res {
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
            _ => {}
        }
        res.expect("Unexpected error during read");
        let f = f32::from_le_bytes(f_buffer);
        results.push(f)
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        let x = nn::silu(&0.);
        assert_eq!(x, 0.)
    }

    #[test]
    fn test_rmsnorm() {
        let u = array![0., 1., 0.];
        let _vv = nn::rmsnorm(&u.view(), &u.view());
    }

    #[test]
    fn test_softmax() {
        let u = array![0., 0., 0.];
        let _v = nn::softmax(&u);

        let u = array![1., 1., 1.];
        let v = nn::softmax(&u);
        assert_eq!(v, array![1. / 3., 1. / 3., 1. / 3.])
    }

    #[test]
    fn test_transformer() {
        let conf = Config {
            dim: 8,
            hidden_dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 4,
            seq_len: 10,
        };
        let mut state = RunState::zeros(&conf);
        let weights = TransformerWeights::zeros(&conf);
        nn::transformer(0, 2, &conf, &mut state, &weights)
    }

    #[test]
    fn test_argmax1() {
        let u = array![0., 1., 0.];
        let v = nn::argmax1(&u).unwrap();
        assert_eq!(v, 1);

        let u = array![1., -1., -99., 100.];
        let v = nn::argmax1(&u).unwrap();
        assert_eq!(v, 3);
    }
}

fn load_tokenizer(path: &str, vocab_size: i32) -> (Vec<(String, f32)>, i32) {
    // load the vocab from the binary file
    let f = File::open(path).expect("Failed to open file");
    let mut input = BufReader::new(f);

    let mut vocab: Vec<(String, f32)> = Vec::new();
    let mut i_buffer = [0u8; std::mem::size_of::<i32>()];
    let mut f_buffer = [0u8; std::mem::size_of::<f32>()];
    let mut s_buffer = [0u8; 1];

    input.read_exact(&mut i_buffer).unwrap();
    let max_token_length = i32::from_le_bytes(i_buffer);

    for _ in 0..vocab_size {
        input.read_exact(&mut f_buffer).unwrap();
        let score = f32::from_le_bytes(f_buffer);

        input.read_exact(&mut i_buffer).unwrap();
        let len = i32::from_le_bytes(i_buffer);

        let mut word: String = String::new();

        for _ in 0..len {
            input.read_exact(&mut s_buffer).unwrap();
            let chr = unsafe { String::from_utf8_unchecked(s_buffer.to_vec()) };
            word.push_str(&chr);
        }

        vocab.push((word, score))
    }

    (vocab, max_token_length)
}

fn tokenize(text: &[u8], vocab: &Vec<(String, f32)>, max_token_length: i32, bos: bool) -> Vec<i32> {
    // tokenize a string with BPE
    // based on leo-du/llama2.rs/blob/master/run.rs#L326
    let mut tokens: Vec<i32> = Vec::new();

    for i in 0..text.len() {
        let char: &str = std::str::from_utf8(&text[i..i + 1]).unwrap();
        let (id, _) = vocab
            .iter()
            .enumerate()
            .find(|x| (*(*x).1).0 == char)
            .expect("illegal character");
        tokens.push(id.try_into().unwrap());
    }

    let mut buffer = String::with_capacity(max_token_length.try_into().unwrap());
    loop {
        let mut best = (-1e10_f32, (usize::MAX, usize::MAX)); // (score, (vocab index, tokens index))

        for i in 0..tokens.len() - 1 {
            buffer.clear();
            buffer.push_str(&(vocab[(tokens[i] as usize)].0));
            buffer.push_str(&(vocab[(tokens[i + 1] as usize)].0));
            if let Some((vid, (_, score))) =
                vocab.iter().enumerate().find(|x| (*(*x).1).0 == buffer)
            {
                if *score > best.0 {
                    best = (*score, (vid, i));
                }
            }
        }

        if best.1 .0 == usize::MAX {
            break; // no more possible merges
        }

        // perform merge
        tokens[best.1 .1] = best.1 .0.try_into().unwrap();
        tokens.remove(best.1 .1 + 1);
    }

    if bos && tokens[0] != 1 {
        tokens.insert(0, 1);
    }

    tokens
}

fn get_next_token(state: &RunState, temperature: f32) -> i32 {
    // get the next token either greedily or with temperature sampling
    if temperature == 0. {
        return nn::argmax1(&state.logits).unwrap();
    }

    let tempered = state.logits.map(|x| x / temperature);
    let probs = nn::softmax(&tempered);
    nn::sample(&probs)
}

fn inference(
    prompt: Vec<i32>,
    steps: i32,
    config: &Config,
    weights: &TransformerWeights,
    vocab: &Vec<(String, f32)>,
    temperature: f32,
) {
    let now = Instant::now();

    let mut iterator = prompt.iter();
    let mut token = *iterator.next().unwrap();
    let mut pos = 0;
    let mut seq: Vec<i32> = vec![token];
    let mut state = RunState::zeros(&config);

    let prompt_len: i32 = prompt.len().try_into().unwrap();
    let max_steps = cmp::min(config.seq_len, prompt_len + steps);

    while pos < max_steps {
        nn::transformer(token, pos, &config, &mut state, &weights);

        token = match iterator.next() {
            Some(t) => *t,
            None => get_next_token(&state, temperature),
        };

        pos += 1;
        seq.push(token);

        print!("{}", vocab[token as usize].0);
        io::stdout().flush().unwrap();
    }

    println!(
        "\nRan at {} tok/s.",
        (seq.len() as f32) / now.elapsed().as_secs_f32()
    );
}

fn main() {
    // run inference on a Llama model

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!(
            "Usage: {} <checkpoint_file> [tokenizer_path] [prompt] [temperature] [max_steps]",
            &args[0]
        );
    }
    let path: String = String::from(&args[1]);
    if !Path::new(&path).exists() {
        panic!("Model path does not exist.");
    };
    let mut input = BufReader::new(File::open(path).expect("Failed to open file"));
    let config = Config::from_binary(&mut input);

    let tok_path: String = match args.get(2) {
        Some(s) => s.to_string(),
        None => String::from("tokenizer.bin"),
    };
    if !Path::new(&tok_path).exists() {
        panic!("Tokenizer path does not exist.");
    };

    let (vocab, max_token_length) = load_tokenizer(&tok_path, config.vocab_size);
    let prompt: Vec<i32> = match args.get(3) {
        Some(s) => tokenize(s.as_bytes(), &vocab, max_token_length, true),
        None => vec![
            1, 3118, 2462, 29892, 263, 2217, 7826, 4257, 365, 2354, 1476, 263, 817, 280, 297, 902,
            5716, 29889, 2296, 6363,
        ],
    };

    let temperature: f32 = match args.get(4) {
        Some(s) => s.parse::<f32>().expect("Failed to parse temperature."),
        None => 0.9,
    };

    let max_steps: i32 = match args.get(5) {
        Some(s) => s.parse::<i32>().expect("Failed to parse max steps."),
        None => 20,
    };

    let weights = TransformerWeights::from_binary(&mut input, &config);
    inference(prompt, max_steps, &config, &weights, &vocab, temperature);
}
