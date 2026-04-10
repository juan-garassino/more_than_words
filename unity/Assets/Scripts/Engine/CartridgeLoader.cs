using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace LivingTales.Engine
{
    /// <summary>
    /// Loads a Living Tales cartridge: spec, tokens, expressions, and the ONNX model path.
    /// Each case/creature is a self-contained cartridge directory.
    /// </summary>
    [Serializable]
    public class TokenDef
    {
        public string id;
        public string token_class;
        public string phase;
        public float[] attractor_weights;
        public string[] affinity_tags;
        public string stream;
        public string agency;
        public bool is_invariant;
        public float temperature;
    }

    [Serializable]
    public class CartridgeSpec
    {
        public string case_id;
        public string title;
        public string mode; // "converging" or "oscillating"
        public int vocab_size;
        public int n_attractor_dims;
        public float convergence_threshold;
        public float convergence_rate;
        public int min_turns;
        public int max_turns;
        public string[] opening_token_ids;
        public string[] invariant_token_ids;
    }

    [Serializable]
    public class ExpressionMap
    {
        public Dictionary<string, string> expressions;
    }

    public class Cartridge
    {
        public CartridgeSpec Spec { get; private set; }
        public List<TokenDef> Tokens { get; private set; }
        public Dictionary<string, string> Expressions { get; private set; }
        public Dictionary<string, int> IdToIdx { get; private set; }
        public Dictionary<int, string> IdxToId { get; private set; }
        public string ModelPath { get; private set; }

        public bool IsCreature => Spec.mode == "oscillating";
        public int NumDimensions => Spec.n_attractor_dims;

        public static Cartridge Load(string cartridgePath)
        {
            var cart = new Cartridge();

            // Load spec
            string specJson = File.ReadAllText(Path.Combine(cartridgePath, "spec.json"));
            cart.Spec = JsonUtility.FromJson<CartridgeSpec>(specJson);

            // Load tokens
            string tokensJson = File.ReadAllText(Path.Combine(cartridgePath, "tokens.json"));
            cart.Tokens = new List<TokenDef>();
            // Unity's JsonUtility doesn't handle top-level arrays, wrap it
            string wrapped = "{\"items\":" + tokensJson + "}";
            var tokenArray = JsonUtility.FromJson<TokenArray>(wrapped);
            cart.Tokens = new List<TokenDef>(tokenArray.items);

            // Load expressions
            cart.Expressions = new Dictionary<string, string>();
            string exprPath = Path.Combine(cartridgePath, "expressions.json");
            if (File.Exists(exprPath))
            {
                string exprJson = File.ReadAllText(exprPath);
                // Parse manually since it's a simple {key: value} map
                cart.Expressions = ParseSimpleJsonMap(exprJson);
            }

            // Build index mappings
            cart.IdToIdx = new Dictionary<string, int>();
            cart.IdxToId = new Dictionary<int, string>();
            for (int i = 0; i < cart.Tokens.Count; i++)
            {
                cart.IdToIdx[cart.Tokens[i].id] = i;
                cart.IdxToId[i] = cart.Tokens[i].id;
            }

            // Model path
            cart.ModelPath = Path.Combine(cartridgePath, "model.onnx");

            Debug.Log($"Loaded cartridge: {cart.Spec.title} ({cart.Spec.vocab_size} tokens, {cart.NumDimensions} dims)");
            return cart;
        }

        public TokenDef GetToken(string id)
        {
            if (IdToIdx.TryGetValue(id, out int idx))
                return Tokens[idx];
            return null;
        }

        public string GetExpression(string tokenId)
        {
            if (Expressions.TryGetValue(tokenId, out string expr))
                return expr;
            // Fallback: humanize the token ID
            string name = tokenId.Contains(":") ? tokenId.Split(':')[1] : tokenId;
            return name.Replace("_", " ");
        }

        public List<TokenDef> GetPlayerTokens()
        {
            var result = new List<TokenDef>();
            foreach (var tok in Tokens)
            {
                if ((tok.agency == "PLAYER" || tok.agency == "SHARED") &&
                    !tok.is_invariant && tok.stream != "OPENING")
                    result.Add(tok);
            }
            return result;
        }

        public List<TokenDef> GetEngineTokens()
        {
            var result = new List<TokenDef>();
            foreach (var tok in Tokens)
            {
                if ((tok.agency == "ENGINE" || tok.agency == "SHARED") &&
                    !tok.is_invariant && tok.stream != "OPENING")
                    result.Add(tok);
            }
            return result;
        }

        // Simple JSON map parser for {"key": "value", ...}
        static Dictionary<string, string> ParseSimpleJsonMap(string json)
        {
            var dict = new Dictionary<string, string>();
            json = json.Trim().TrimStart('{').TrimEnd('}');
            if (string.IsNullOrWhiteSpace(json)) return dict;

            // Basic parser — works for simple string:string maps
            bool inKey = false, inValue = false;
            string currentKey = "", currentValue = "";
            bool escaped = false;

            int state = 0; // 0=seekKey, 1=inKey, 2=seekColon, 3=seekValue, 4=inValue
            foreach (char c in json)
            {
                if (escaped) { if (state == 1) currentKey += c; else if (state == 4) currentValue += c; escaped = false; continue; }
                if (c == '\\') { escaped = true; continue; }

                switch (state)
                {
                    case 0: if (c == '"') { state = 1; currentKey = ""; } break;
                    case 1: if (c == '"') state = 2; else currentKey += c; break;
                    case 2: if (c == ':') state = 3; break;
                    case 3: if (c == '"') { state = 4; currentValue = ""; } break;
                    case 4:
                        if (c == '"')
                        {
                            dict[currentKey] = currentValue;
                            state = 0;
                        }
                        else currentValue += c;
                        break;
                }
            }
            return dict;
        }
    }

    // Helper for deserializing JSON arrays in Unity
    [Serializable]
    public class TokenArray
    {
        public TokenDef[] items;
    }
}
