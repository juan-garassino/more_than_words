using System;
using System.Collections.Generic;
using UnityEngine;


namespace LivingTales.Engine
{
    /// <summary>
    /// Runs SceneTransformer ONNX inference via Unity Sentis.
    ///
    /// One forward pass → N token predictions (one per attractor dimension).
    /// Each head outputs logits over the vocabulary, masked to dimension-relevant tokens.
    ///
    /// Usage:
    ///   var runtime = new SceneTransformerRuntime(cartridge);
    ///   runtime.LoadModel("path/to/model.onnx");
    ///   var scene = runtime.PredictScene(contextTokenIds, temperature: 0.8f);
    ///   // scene = [(headIdx, tokenIdx, tokenId, probability), ...]
    /// </summary>
    public class SceneTransformerRuntime : IDisposable
    {
        public struct ScenePrediction
        {
            public int headIndex;
            public int tokenIndex;
            public string tokenId;
            public float probability;
        }

        private Cartridge cartridge;
        private Unity.InferenceEngine.Model model;
        private Unity.InferenceEngine.Worker worker;
        private bool isLoaded = false;

        // Encoding maps (mirroring Python's _build_mappings)
        private Dictionary<string, int> classToIdx = new();
        private Dictionary<string, int> phaseToIdx = new();
        private Dictionary<string, int> streamToIdx = new();
        private Dictionary<string, int> agencyToIdx = new();

        // Head vocab masks (built from attractor weights)
        private bool[][] headMasks; // [nHeads][vocabSize]

        public int NumHeads => cartridge.NumDimensions;
        public bool IsLoaded => isLoaded;

        public SceneTransformerRuntime(Cartridge cartridge)
        {
            this.cartridge = cartridge;
            BuildEncodingMaps();
            BuildHeadMasks();
        }

        /// <summary>
        /// Load the ONNX model.
        /// </summary>
        public void LoadModel(string onnxPath)
        {
            try
            {
                model = Unity.InferenceEngine.ModelLoader.Load(onnxPath);
                worker = new Unity.InferenceEngine.Worker(model, Unity.InferenceEngine.BackendType.CPU);
                isLoaded = true;
                Debug.Log($"SceneTransformer loaded: {onnxPath} ({NumHeads} heads)");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to load model: {e.Message}");
                isLoaded = false;
            }
        }

        /// <summary>
        /// Run inference: predict a full scene (N tokens) from the context.
        /// </summary>
        public List<ScenePrediction> PredictScene(int[] contextTokenIds, float temperature = 0.8f)
        {
            if (!isLoaded)
            {
                Debug.LogWarning("Model not loaded, returning empty scene");
                return new List<ScenePrediction>();
            }

            int seqLen = contextTokenIds.Length;

            // Build input tensors
            var tokenIdsTensor = BuildIntTensor(contextTokenIds, seqLen);
            var classIdsTensor = BuildClassIds(contextTokenIds, seqLen);
            var phaseIdsTensor = BuildPhaseIds(contextTokenIds, seqLen);
            var streamIdsTensor = BuildStreamIds(contextTokenIds, seqLen);
            var agencyIdsTensor = BuildAgencyIds(contextTokenIds, seqLen);
            var paddingTensor = BuildPaddingMask(seqLen);

            // Set inputs
            worker.SetInput("token_ids", tokenIdsTensor);
            worker.SetInput("class_ids", classIdsTensor);
            worker.SetInput("phase_ids", phaseIdsTensor);
            worker.SetInput("stream_ids", streamIdsTensor);
            worker.SetInput("agency_ids", agencyIdsTensor);
            worker.SetInput("padding_mask", paddingTensor);

            // Execute
            worker.Schedule();

            // Read outputs — one per head
            var results = new List<ScenePrediction>();

            for (int d = 0; d < NumHeads; d++)
            {
                string outputName = $"head_{d}";
                var output = worker.PeekOutput(outputName);

                // Output shape: (1, seqLen, vocabSize)
                // Take last position — download tensor to CPU array
                int vocabSize = cartridge.Spec.vocab_size;
                float[] logits = new float[vocabSize];
                var outputData = output.ToReadOnlyArray();
                int lastPosOffset = (seqLen - 1) * vocabSize;

                for (int v = 0; v < vocabSize; v++)
                {
                    logits[v] = outputData[lastPosOffset + v];

                    // Apply head mask
                    if (!headMasks[d][v])
                        logits[v] = float.NegativeInfinity;
                }

                // Sample with temperature
                int chosenIdx = SampleFromLogits(logits, temperature, out float prob);

                if (chosenIdx >= 0 && cartridge.IdxToId.TryGetValue(chosenIdx, out string tokenId))
                {
                    results.Add(new ScenePrediction
                    {
                        headIndex = d,
                        tokenIndex = chosenIdx,
                        tokenId = tokenId,
                        probability = prob,
                    });
                }
            }

            // Dispose tensors
            tokenIdsTensor.Dispose();
            classIdsTensor.Dispose();
            phaseIdsTensor.Dispose();
            streamIdsTensor.Dispose();
            agencyIdsTensor.Dispose();
            paddingTensor.Dispose();

            return results;
        }

        /// <summary>
        /// Sample a token index from logits using temperature scaling.
        /// </summary>
        private int SampleFromLogits(float[] logits, float temperature, out float probability)
        {
            probability = 0f;
            int n = logits.Length;

            // Find max for numerical stability
            float maxLogit = float.NegativeInfinity;
            for (int i = 0; i < n; i++)
                if (logits[i] > maxLogit) maxLogit = logits[i];

            if (float.IsNegativeInfinity(maxLogit))
                return -1; // No valid tokens

            // Compute softmax with temperature
            float[] probs = new float[n];
            float sum = 0f;

            for (int i = 0; i < n; i++)
            {
                if (float.IsNegativeInfinity(logits[i]))
                {
                    probs[i] = 0f;
                }
                else
                {
                    probs[i] = Mathf.Exp((logits[i] - maxLogit) / Mathf.Max(temperature, 0.01f));
                    sum += probs[i];
                }
            }

            if (sum < 1e-12f)
                return -1;

            // Normalize
            for (int i = 0; i < n; i++)
                probs[i] /= sum;

            // Sample
            float r = UnityEngine.Random.value;
            float cumulative = 0f;
            for (int i = 0; i < n; i++)
            {
                cumulative += probs[i];
                if (r <= cumulative)
                {
                    probability = probs[i];
                    return i;
                }
            }

            // Fallback: return highest probability
            int best = 0;
            for (int i = 1; i < n; i++)
                if (probs[i] > probs[best]) best = i;
            probability = probs[best];
            return best;
        }

        // ─────────────────────────────────────────────────────────────
        // Tensor builders
        // ─────────────────────────────────────────────────────────────

        private Unity.InferenceEngine.Tensor<int> BuildIntTensor(int[] values, int seqLen)
        {
            var tensor = new Unity.InferenceEngine.Tensor<int>(new Unity.InferenceEngine.TensorShape(1, seqLen));
            for (int i = 0; i < seqLen; i++)
                tensor[0, i] = values[i];
            return tensor;
        }

        private Unity.InferenceEngine.Tensor<int> BuildClassIds(int[] tokenIds, int seqLen)
        {
            var tensor = new Unity.InferenceEngine.Tensor<int>(new Unity.InferenceEngine.TensorShape(1, seqLen));
            for (int i = 0; i < seqLen; i++)
            {
                if (cartridge.IdxToId.TryGetValue(tokenIds[i], out string id))
                {
                    var tok = cartridge.GetToken(id);
                    if (tok != null && classToIdx.TryGetValue(tok.token_class, out int idx))
                        tensor[0, i] = idx;
                }
            }
            return tensor;
        }

        private Unity.InferenceEngine.Tensor<int> BuildPhaseIds(int[] tokenIds, int seqLen)
        {
            var tensor = new Unity.InferenceEngine.Tensor<int>(new Unity.InferenceEngine.TensorShape(1, seqLen));
            for (int i = 0; i < seqLen; i++)
            {
                if (cartridge.IdxToId.TryGetValue(tokenIds[i], out string id))
                {
                    var tok = cartridge.GetToken(id);
                    if (tok != null && phaseToIdx.TryGetValue(tok.phase, out int idx))
                        tensor[0, i] = idx;
                }
            }
            return tensor;
        }

        private Unity.InferenceEngine.Tensor<int> BuildStreamIds(int[] tokenIds, int seqLen)
        {
            var tensor = new Unity.InferenceEngine.Tensor<int>(new Unity.InferenceEngine.TensorShape(1, seqLen));
            for (int i = 0; i < seqLen; i++)
            {
                if (cartridge.IdxToId.TryGetValue(tokenIds[i], out string id))
                {
                    var tok = cartridge.GetToken(id);
                    if (tok != null && streamToIdx.TryGetValue(tok.stream, out int idx))
                        tensor[0, i] = idx;
                }
            }
            return tensor;
        }

        private Unity.InferenceEngine.Tensor<int> BuildAgencyIds(int[] tokenIds, int seqLen)
        {
            var tensor = new Unity.InferenceEngine.Tensor<int>(new Unity.InferenceEngine.TensorShape(1, seqLen));
            for (int i = 0; i < seqLen; i++)
            {
                if (cartridge.IdxToId.TryGetValue(tokenIds[i], out string id))
                {
                    var tok = cartridge.GetToken(id);
                    if (tok != null && agencyToIdx.TryGetValue(tok.agency, out int idx))
                        tensor[0, i] = idx;
                }
            }
            return tensor;
        }

        private Unity.InferenceEngine.Tensor<int> BuildPaddingMask(int seqLen)
        {
            // No padding — all positions are valid
            var tensor = new Unity.InferenceEngine.Tensor<int>(new Unity.InferenceEngine.TensorShape(1, seqLen));
            // 0 = not padded (Sentis uses int, not bool)
            return tensor;
        }

        // ─────────────────────────────────────────────────────────────
        // Encoding maps
        // ─────────────────────────────────────────────────────────────

        private void BuildEncodingMaps()
        {
            // Must match Python's TokenClass, TokenPhase, TokenStream, TokenAgency enums
            string[] classes = { "SUSPECT", "MOTIVE", "EVENT", "LOCATION", "OBJECT",
                                "ACTION", "EMOTION", "MODIFIER", "WITNESS", "TIME",
                                "ACCOMPLICE", "UNKNOWN" };
            for (int i = 0; i < classes.Length; i++)
                classToIdx[classes[i]] = i;

            string[] phases = { "EARLY", "MID", "LATE", "INVARIANT" };
            for (int i = 0; i < phases.Length; i++)
                phaseToIdx[phases[i]] = i;

            string[] streams = { "EVIDENCE", "ATMOSPHERE", "OPENING", "INVARIANT" };
            for (int i = 0; i < streams.Length; i++)
                streamToIdx[streams[i]] = i;

            string[] agencies = { "PLAYER", "ENGINE", "SHARED" };
            for (int i = 0; i < agencies.Length; i++)
                agencyToIdx[agencies[i]] = i;
        }

        private void BuildHeadMasks()
        {
            int nDims = cartridge.NumDimensions;
            int vocabSize = cartridge.Spec.vocab_size;
            float threshold = 0.05f;

            headMasks = new bool[nDims][];
            for (int d = 0; d < nDims; d++)
            {
                headMasks[d] = new bool[vocabSize];
                int count = 0;

                for (int t = 0; t < cartridge.Tokens.Count; t++)
                {
                    var tok = cartridge.Tokens[t];
                    if (tok.is_invariant || tok.stream == "OPENING")
                        continue;

                    if (d < tok.attractor_weights.Length &&
                        Mathf.Abs(tok.attractor_weights[d]) > threshold)
                    {
                        headMasks[d][t] = true;
                        count++;
                    }
                }

                // Fallback: if too few candidates, take top 10
                if (count < 3)
                {
                    var scores = new List<(int idx, float score)>();
                    for (int t = 0; t < cartridge.Tokens.Count; t++)
                    {
                        if (!cartridge.Tokens[t].is_invariant &&
                            d < cartridge.Tokens[t].attractor_weights.Length)
                            scores.Add((t, Mathf.Abs(cartridge.Tokens[t].attractor_weights[d])));
                    }
                    scores.Sort((a, b) => b.score.CompareTo(a.score));
                    for (int i = 0; i < Mathf.Min(10, scores.Count); i++)
                        headMasks[d][scores[i].idx] = true;
                }

                Debug.Log($"Head {d}: {count} tokens in mask");
            }
        }

        public void Dispose()
        {
            worker?.Dispose();
            // Model is a data class in Sentis 2.2+, no Dispose needed
        }
    }
}
