enum NormType {
    LayerNorm,
    RMSNorm,
}

enum PosEncoding {
    Learned,
    RoPE {
        theta: f32,
    },
}

enum FfnType {
    Standard,
    Gated,
}

enum Activation {
    Gelu,
    Silu,
}

pub struct DecoderConfig {
    vocab_size: u32,
    hidden_size: u32,
    num_layers: u32,
    num_attention_heads: u32,
    num_kv_heads: u32,
    intermediate_size: u32,
    max_position_embeddings: u32,
    norm_type: NormType,
    norm_eps: f32,
    pos_encoding: PosEncoding,
    ffn_type: FfnType,
    activation: Activation,
    use_bias: bool,
    tie_word_embeddings: bool,
}

impl DecoderConfig {
    pub fn new(json: &str) -> Self {
        let v: serde_json::Value = serde_json::from_str(json)
            .expect("Failed to parse config JSON");
        let model_type = v["model_type"].as_str().unwrap_or_else(|| panic!("model_type is required in config"));

        let vocab_size = match model_type {
            "gpt2" => v["vocab_size"].as_u64().unwrap() as u32,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let hidden_size = match model_type {
            "gpt2" => v["n_embd"].as_u64().unwrap() as u32,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let num_layers = match model_type {
            "gpt2" => v["n_layer"].as_u64().unwrap() as u32,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let num_attention_heads = match model_type {
            "gpt2" => v["n_head"].as_u64().unwrap() as u32,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let num_kv_heads = match model_type {
            "gpt2" => num_attention_heads,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let intermediate_size = match model_type {
            "gpt2" => if v["n_inner"].as_u64().is_some() {
                v["n_inner"].as_u64().unwrap() as u32
            } else {
                4 * hidden_size
            },
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let max_position_embeddings = match model_type {
            "gpt2" => v["n_positions"].as_u64().unwrap() as u32,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let norm_type = match model_type {
            "gpt2" => NormType::LayerNorm,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let norm_eps = match model_type {
            "gpt2" => v["layer_norm_epsilon"].as_f64().unwrap() as f32,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let pos_encoding = match model_type {
            "gpt2" => PosEncoding::Learned,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let ffn_type = match model_type {
            "gpt2" => FfnType::Standard,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let activation = match model_type {
            "gpt2" => Activation::Gelu,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let use_bias = match model_type {
            "gpt2" => true,
            _ => panic!("Unsupported model type: {}", model_type),
        };

        let tie_word_embeddings = match model_type {
            "gpt2" => v["tie_word_embeddings"].as_bool().unwrap_or(true),
            _ => panic!("Unsupported model type: {}", model_type),
        };

        Self {
            vocab_size,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            intermediate_size,
            max_position_embeddings,
            norm_type,
            norm_eps,
            pos_encoding,
            ffn_type,
            activation,
            use_bias,
            tie_word_embeddings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_config() {
        let json = include_str!("../tests/fixtures/test_config.json");
        let config = DecoderConfig::new(json);
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_kv_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.max_position_embeddings, 1024);
        match config.norm_type {
            NormType::LayerNorm => {},
            _ => panic!("Expected LayerNorm"),
        }
        assert_eq!(config.norm_eps, 1e-5);
        match config.pos_encoding {
            PosEncoding::Learned => {},
            _ => panic!("Expected Learned Positional Encoding"),
        }
        match config.ffn_type {
            FfnType::Standard => {},
            _ => panic!("Expected Standard FFN"),
        }
        match config.activation {
            Activation::Gelu => {},
            _ => panic!("Expected Gelu activation"),
        }
        assert!(config.use_bias);
        assert!(config.tie_word_embeddings);
    }
}
    