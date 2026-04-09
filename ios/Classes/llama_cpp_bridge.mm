/*
 * Flutter Llama - llama.cpp Bridge for iOS
 * 
 * This file provides a C++ bridge between Swift and llama.cpp
 * Updated for latest llama.cpp API
 */

#import <Foundation/Foundation.h>
#include <string>
#include <vector>
#include <mutex>

// Include llama.cpp headers
#include "../../llama.cpp/include/llama.h"

// Global state
static llama_model* g_model = nullptr;
static llama_context* g_context = nullptr;
static const llama_vocab* g_vocab = nullptr;
static llama_sampler* g_sampler = nullptr;
static std::mutex g_mutex;
static bool g_should_stop = false;
static std::vector<std::string> g_stream_tokens;
static size_t g_stream_pos = 0;

extern "C" {

// Initialize and load model
bool llama_init_model(
    const char* model_path,
    int32_t n_threads,
    int32_t n_gpu_layers,
    int32_t context_size,
    int32_t batch_size,
    bool use_gpu,
    bool verbose
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    NSLog(@"[llama_cpp_bridge] Initializing model: %s", model_path);
    NSLog(@"[llama_cpp_bridge] Threads: %d, GPU layers: %d, Context: %d", 
          n_threads, n_gpu_layers, context_size);
    
    // Free existing model if any
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    
    // Load dynamic backends
    ggml_backend_load_all();
    
    // Set up model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = use_gpu ? n_gpu_layers : 0;
    
    // Load model
    g_model = llama_model_load_from_file(model_path, model_params);
    if (!g_model) {
        NSLog(@"[llama_cpp_bridge] Failed to load model from: %s", model_path);
        return false;
    }
    
    // Get vocab
    g_vocab = llama_model_get_vocab(g_model);
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size;
    ctx_params.n_batch = batch_size;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    
    g_context = llama_init_from_model(g_model, ctx_params);
    if (!g_context) {
        NSLog(@"[llama_cpp_bridge] Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return false;
    }
    
    // Initialize sampler chain
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    g_sampler = llama_sampler_chain_init(sparams);
    
    // Add samplers
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));
    
    NSLog(@"[llama_cpp_bridge] Model loaded successfully");
    NSLog(@"[llama_cpp_bridge] Context size: %d", llama_n_ctx(g_context));
    
    return true;
}

// Generate text
bool llama_generate(
    const char* prompt,
    float temperature,
    float top_p,
    int32_t top_k,
    int32_t max_tokens,
    float repeat_penalty,
    char* output,
    int32_t output_size,
    int32_t* tokens_generated
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (!g_model || !g_context || !g_vocab) {
        NSLog(@"[llama_cpp_bridge] Model not loaded");
        return false;
    }
    
    NSLog(@"[llama_cpp_bridge] Generating with prompt: %.50s...", prompt);
    
    std::string prompt_text(prompt);
    
    // Tokenize prompt
    const int n_prompt = -llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    
    if (llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        NSLog(@"[llama_cpp_bridge] Failed to tokenize prompt");
        return false;
    }
    
    // Create batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    
    // Decode prompt
    if (llama_decode(g_context, batch) != 0) {
        NSLog(@"[llama_cpp_bridge] Failed to decode prompt");
        return false;
    }
    
    // Update sampler with new parameters
    llama_sampler_free(g_sampler);
    
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));
    
    // Generate tokens
    std::string result;
    int n_gen = 0;
    int n_pos = prompt_tokens.size();
    
    g_should_stop = false;
    
    for (int i = 0; i < max_tokens; i++) {
        if (g_should_stop) {
            NSLog(@"[llama_cpp_bridge] Generation stopped by user");
            break;
        }
        
        // Sample next token
        llama_token new_token = llama_sampler_sample(g_sampler, g_context, -1);
        
        // Check for EOS
        if (llama_vocab_is_eog(g_vocab, new_token)) {
            NSLog(@"[llama_cpp_bridge] EOS token reached");
            break;
        }
        
        // Convert token to text
        char token_str[256] = {0};
        int n = llama_token_to_piece(g_vocab, new_token, token_str, sizeof(token_str) - 1, 0, true);
        if (n > 0) {
            token_str[n] = '\0';
            result.append(token_str);
        }
        
        // Prepare for next iteration
        batch = llama_batch_get_one(&new_token, 1);
        n_pos++;
        
        if (llama_decode(g_context, batch) != 0) {
            NSLog(@"[llama_cpp_bridge] Failed to decode token");
            break;
        }
        
        n_gen++;
    }
    
    // Copy result
    size_t copy_len = std::min(result.length(), (size_t)(output_size - 1));
    memcpy(output, result.c_str(), copy_len);
    output[copy_len] = '\0';
    *tokens_generated = n_gen;
    
    NSLog(@"[llama_cpp_bridge] Generated %d tokens", n_gen);
    return true;
}

// Initialize streaming generation
void llama_generate_stream_init(
    const char* prompt,
    float temperature,
    float top_p,
    int32_t top_k,
    int32_t max_tokens,
    float repeat_penalty
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    NSLog(@"[llama_cpp_bridge] Initializing stream generation");
    
    if (!g_model || !g_context || !g_vocab) {
        NSLog(@"[llama_cpp_bridge] Model not loaded");
        return;
    }
    
    g_should_stop = false;
    g_stream_tokens.clear();
    g_stream_pos = 0;
    
    std::string prompt_text(prompt);
    
    // Tokenize prompt
    const int n_prompt = -llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    
    if (llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        NSLog(@"[llama_cpp_bridge] Failed to tokenize prompt");
        return;
    }
    
    // Create batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    
    // Decode prompt
    if (llama_decode(g_context, batch) != 0) {
        NSLog(@"[llama_cpp_bridge] Failed to decode prompt");
        return;
    }
    
    // Update sampler
    if (g_sampler) {
        llama_sampler_free(g_sampler);
    }
    
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));
    
    // Pre-generate tokens and convert to strings
    int n_pos = prompt_tokens.size();
    for (int i = 0; i < max_tokens; i++) {
        if (g_should_stop) break;
        
        llama_token new_token = llama_sampler_sample(g_sampler, g_context, -1);
        
        if (llama_vocab_is_eog(g_vocab, new_token)) {
            break;
        }
        
        // Convert token to text and store
        char token_str[256] = {0};
        int n = llama_token_to_piece(g_vocab, new_token, token_str, sizeof(token_str) - 1, 0, true);
        if (n > 0) {
            token_str[n] = '\0';
            g_stream_tokens.push_back(std::string(token_str));
        }
        
        batch = llama_batch_get_one(&new_token, 1);
        n_pos++;
        
        if (llama_decode(g_context, batch) != 0) {
            break;
        }
    }
    
    NSLog(@"[llama_cpp_bridge] Pre-generated %zu tokens for streaming", g_stream_tokens.size());
}

// Get next token in stream
bool llama_generate_stream_next(
    char* output,
    int32_t output_size
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_should_stop || g_stream_pos >= g_stream_tokens.size()) {
        return false;
    }
    
    const std::string& token = g_stream_tokens[g_stream_pos++];
    
    size_t copy_len = std::min(token.length(), (size_t)(output_size - 1));
    memcpy(output, token.c_str(), copy_len);
    output[copy_len] = '\0';
    
    return true;
}

// End streaming generation
void llama_generate_stream_end() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    NSLog(@"[llama_cpp_bridge] Ending stream generation");
    g_stream_tokens.clear();
    g_stream_pos = 0;
}

// Get model information
void llama_get_model_info(
    int64_t* n_params,
    int32_t* n_layers,
    int32_t* context_size
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (!g_model || !g_context) {
        *n_params = 0;
        *n_layers = 0;
        *context_size = 0;
        return;
    }
    
    *n_params = llama_model_n_params(g_model);
    *n_layers = llama_model_n_layer(g_model);
    *context_size = llama_n_ctx(g_context);
}

// Free model
void llama_cpp_bridge_free_model() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    NSLog(@"[llama_cpp_bridge] Freeing model");
    
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    
    g_vocab = nullptr;
    
    NSLog(@"[llama_cpp_bridge] Model freed successfully");
}

// Stop generation
void llama_stop_generation() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    NSLog(@"[llama_cpp_bridge] Stopping generation");
    g_should_stop = true;
}

// ──────────────────────────────────────────────────────
// Multimodal (Vision) support via mtmd
// ──────────────────────────────────────────────────────

} // extern "C" (text-only section)

#include "mtmd.h"
#include "mtmd-helper.h"

// Multimodal global state
static mtmd_context* g_mtmd_ctx = nullptr;
static std::mutex g_mtmd_mutex;

extern "C" {

bool llama_init_multimodal(
    const char* mmproj_path
) {
    std::lock_guard<std::mutex> lock(g_mtmd_mutex);

    if (!g_model) {
        NSLog(@"[llama_cpp_bridge] Cannot init multimodal: text model not loaded");
        return false;
    }

    if (g_mtmd_ctx) {
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
    }

    NSLog(@"[llama_cpp_bridge] Initializing multimodal with mmproj: %s", mmproj_path);

    auto params = mtmd_context_params_default();
    params.use_gpu = true;
    params.n_threads = 4;
    params.warmup = true;

    g_mtmd_ctx = mtmd_init_from_file(mmproj_path, g_model, params);
    if (!g_mtmd_ctx) {
        NSLog(@"[llama_cpp_bridge] Failed to init multimodal context");
        return false;
    }

    bool has_vision = mtmd_support_vision(g_mtmd_ctx);
    bool has_audio = mtmd_support_audio(g_mtmd_ctx);
    NSLog(@"[llama_cpp_bridge] Multimodal initialized: vision=%d, audio=%d", has_vision, has_audio);

    return true;
}

bool llama_generate_with_image(
    const char* prompt,
    const unsigned char* image_data,
    int32_t image_len,
    float temperature,
    int32_t max_tokens,
    char* output,
    int32_t output_size,
    int32_t* tokens_generated
) {
    std::lock_guard<std::mutex> lock(g_mtmd_mutex);

    if (!g_model || !g_context || !g_mtmd_ctx || !g_vocab) {
        NSLog(@"[llama_cpp_bridge] Multimodal not ready");
        return false;
    }

    NSLog(@"[llama_cpp_bridge] Generating with image: prompt=%.50s..., image=%d bytes", prompt, image_len);

    // 1. Create bitmap from image buffer
    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(g_mtmd_ctx, image_data, (size_t)image_len);
    if (!bmp) {
        NSLog(@"[llama_cpp_bridge] Failed to create bitmap from image data");
        return false;
    }

    // 2. Tokenize prompt + image
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    mtmd_input_text text_input = { prompt, true, true };
    const mtmd_bitmap* bitmaps[] = { bmp };

    int32_t ret = mtmd_tokenize(g_mtmd_ctx, chunks, &text_input, bitmaps, 1);
    mtmd_bitmap_free(bmp);

    if (ret != 0) {
        NSLog(@"[llama_cpp_bridge] Tokenize failed: %d", ret);
        mtmd_input_chunks_free(chunks);
        return false;
    }

    // 3. Clear KV cache and evaluate all chunks (text + image embeddings)
    llama_kv_cache_clear(g_context);

    llama_pos n_past = 0;
    ret = mtmd_helper_eval_chunks(g_mtmd_ctx, g_context, chunks, n_past, 0,
                                  512, true, &n_past);
    mtmd_input_chunks_free(chunks);

    if (ret != 0) {
        NSLog(@"[llama_cpp_bridge] Eval chunks failed: %d", ret);
        return false;
    }

    // 4. Update sampler with generation parameters
    llama_sampler_free(g_sampler);
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));

    // 5. Generate tokens
    std::string result;
    int n_gen = 0;
    g_should_stop = false;

    for (int i = 0; i < max_tokens; i++) {
        if (g_should_stop) break;

        llama_token new_token = llama_sampler_sample(g_sampler, g_context, -1);

        if (llama_vocab_is_eog(g_vocab, new_token)) {
            NSLog(@"[llama_cpp_bridge] Vision: EOS reached");
            break;
        }

        char token_str[256] = {0};
        int n = llama_token_to_piece(g_vocab, new_token, token_str, sizeof(token_str) - 1, 0, true);
        if (n > 0) {
            token_str[n] = '\0';
            result.append(token_str);
        }

        llama_batch batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_context, batch) != 0) {
            NSLog(@"[llama_cpp_bridge] Vision: decode failed at token %d", i);
            break;
        }
        n_gen++;
    }

    size_t copy_len = std::min(result.length(), (size_t)(output_size - 1));
    memcpy(output, result.c_str(), copy_len);
    output[copy_len] = '\0';
    *tokens_generated = n_gen;

    NSLog(@"[llama_cpp_bridge] Vision: generated %d tokens", n_gen);
    return true;
}

void llama_free_multimodal() {
    std::lock_guard<std::mutex> lock(g_mtmd_mutex);
    if (g_mtmd_ctx) {
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
        NSLog(@"[llama_cpp_bridge] Multimodal context freed");
    }
}

bool llama_multimodal_supports_vision() {
    std::lock_guard<std::mutex> lock(g_mtmd_mutex);
    return g_mtmd_ctx && mtmd_support_vision(g_mtmd_ctx);
}

} // extern "C"
