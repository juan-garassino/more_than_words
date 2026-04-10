using System.Collections.Generic;
using UnityEngine;

namespace LivingTales.Engine
{
    /// <summary>
    /// Manages the token context window, placed tokens, phase gating,
    /// and dimension tracking for the game loop.
    /// </summary>
    public class TokenManager
    {
        public Cartridge Cartridge { get; private set; }

        // Rolling context window (token indices)
        private List<int> contextIndices = new();
        private List<string> contextIds = new();
        private HashSet<string> placedIds = new();
        private int maxContextLength = 64;
        private int dialoguePosition = 0;

        // Dimension tracking
        public float[] DimensionValues { get; private set; }
        public float ConvergenceScore => MinDimension();

        // Phase mapping
        private Dictionary<string, int> phaseToMinTurn = new()
        {
            { "EARLY", 0 },
            { "MID", 4 },
            { "LATE", 8 },
            { "INVARIANT", 999 },
        };

        public TokenManager(Cartridge cartridge, int contextWindow = 64)
        {
            Cartridge = cartridge;
            maxContextLength = contextWindow;
            DimensionValues = new float[cartridge.NumDimensions];

            if (cartridge.IsCreature)
            {
                // Creatures start with slight initial values
                for (int i = 0; i < DimensionValues.Length; i++)
                    DimensionValues[i] = 0.15f;
            }
        }

        /// <summary>
        /// Place opening tokens at the start of the game.
        /// </summary>
        public List<TokenDef> PlaceOpeningTokens()
        {
            var placed = new List<TokenDef>();
            foreach (string tid in Cartridge.Spec.opening_token_ids)
            {
                var tok = Cartridge.GetToken(tid);
                if (tok == null) continue;

                PlaceToken(tok);
                placed.Add(tok);
            }
            return placed;
        }

        /// <summary>
        /// Place a token into the context.
        /// </summary>
        public void PlaceToken(TokenDef token)
        {
            int idx = Cartridge.IdToIdx[token.id];
            contextIndices.Add(idx);
            contextIds.Add(token.id);
            placedIds.Add(token.id);
            dialoguePosition++;

            // Apply attractor weights to dimensions
            ApplyWeights(token);

            // Rolling window: trim if too long
            if (contextIndices.Count > maxContextLength)
            {
                contextIndices.RemoveAt(0);
                contextIds.RemoveAt(0);
            }
        }

        /// <summary>
        /// Apply a token's attractor weights to the dimension values.
        /// </summary>
        private void ApplyWeights(TokenDef token)
        {
            float rate = Cartridge.Spec.convergence_rate;
            for (int d = 0; d < DimensionValues.Length; d++)
            {
                if (d < token.attractor_weights.Length)
                {
                    float newVal = DimensionValues[d] + token.attractor_weights[d] * rate;
                    // Converging mode: dimensions only go up
                    if (!Cartridge.IsCreature)
                        newVal = Mathf.Max(DimensionValues[d], newVal);
                    DimensionValues[d] = Mathf.Clamp(newVal, 0f, 1f);
                }
            }
        }

        /// <summary>
        /// Apply natural decay for creatures (call each turn).
        /// </summary>
        public void ApplyDecay(float amount = 0.01f)
        {
            if (!Cartridge.IsCreature) return;
            for (int d = 0; d < DimensionValues.Length; d++)
                DimensionValues[d] = Mathf.Max(0f, DimensionValues[d] - amount);
        }

        /// <summary>
        /// Reset placed tokens for creature replay (every N turns).
        /// </summary>
        public void ResetPlacedForCreature()
        {
            placedIds.Clear();
            foreach (string tid in Cartridge.Spec.opening_token_ids)
                placedIds.Add(tid);
        }

        /// <summary>
        /// Get available player tokens (phase-gated, not yet placed).
        /// </summary>
        public List<TokenDef> GetAvailablePlayerTokens()
        {
            var available = new List<TokenDef>();
            int gameTurn = dialoguePosition / 2;

            foreach (var tok in Cartridge.GetPlayerTokens())
            {
                if (placedIds.Contains(tok.id) && !Cartridge.IsCreature)
                    continue;
                if (!IsPhaseValid(tok, gameTurn))
                    continue;
                available.Add(tok);
            }

            // Creature: if no tokens available, reset
            if (available.Count == 0 && Cartridge.IsCreature)
            {
                ResetPlacedForCreature();
                return GetAvailablePlayerTokens();
            }

            return available;
        }

        /// <summary>
        /// Check if a token's phase is valid at the current game turn.
        /// </summary>
        public bool IsPhaseValid(TokenDef token, int gameTurn)
        {
            if (!phaseToMinTurn.TryGetValue(token.phase, out int minTurn))
                return true;
            return gameTurn >= minTurn;
        }

        /// <summary>
        /// Check if the game has converged (mystery solved).
        /// </summary>
        public bool IsConverged()
        {
            int gameTurn = dialoguePosition / 2;
            return ConvergenceScore >= Cartridge.Spec.convergence_threshold &&
                   gameTurn >= Cartridge.Spec.min_turns;
        }

        /// <summary>
        /// Get the current context as token index arrays for model input.
        /// </summary>
        public int[] GetContextTokenIds() => contextIndices.ToArray();
        public string[] GetContextStringIds() => contextIds.ToArray();
        public int ContextLength => contextIndices.Count;

        private float MinDimension()
        {
            if (DimensionValues.Length == 0) return 0f;
            float min = DimensionValues[0];
            for (int i = 1; i < DimensionValues.Length; i++)
                if (DimensionValues[i] < min) min = DimensionValues[i];
            return min;
        }
    }
}
