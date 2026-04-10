using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LivingTales.Engine;

namespace LivingTales.Game
{
    /// <summary>
    /// Main game loop for Living Tales.
    ///
    /// Flow:
    ///   1. Load cartridge (case JSON + ONNX model)
    ///   2. Place opening tokens
    ///   3. Show available player actions
    ///   4. Player taps one → add to context
    ///   5. Run SceneTransformer → get N tokens
    ///   6. AnimationRouter plays scene
    ///   7. Update dimensions, check convergence
    ///   8. Loop (creature: infinite, mystery: until solved)
    /// </summary>
    public class GameLoop : MonoBehaviour
    {
        [Header("Configuration")]
        public string cartridgePath = "Cartridges/little_creature_M";
        public float turnDecay = 0.01f;         // dimension decay per turn (creatures only)
        public float sceneStaggerDelay = 0.3f;   // delay between scene token animations
        public int creatureResetInterval = 20;    // reset placed tokens every N turns

        [Header("References")]
        public AnimationRouter animationRouter;
        // public SceneTransformerRuntime modelRuntime;  // TODO: wire up ONNX runtime
        // public ActionBar actionBar;                    // TODO: wire up UI
        // public NarrativeOverlay narrativeOverlay;       // TODO: wire up UI

        // Internal state
        private Cartridge cartridge;
        private TokenManager tokenManager;
        private int turnCount = 0;
        private bool gameActive = false;
        private bool isPlayerTurn = true;

        void Start()
        {
            StartGame();
        }

        /// <summary>
        /// Initialize the game with a cartridge.
        /// </summary>
        public void StartGame()
        {
            string fullPath = Path.Combine(Application.streamingAssetsPath, cartridgePath);
            cartridge = Cartridge.Load(fullPath);
            tokenManager = new TokenManager(cartridge);

            // Place opening tokens
            var openingTokens = tokenManager.PlaceOpeningTokens();
            foreach (var tok in openingTokens)
            {
                string expr = cartridge.GetExpression(tok.id);
                Debug.Log($"Opening: {tok.id} — {expr}");
                // narrativeOverlay?.Show(expr);
            }

            gameActive = true;
            isPlayerTurn = true;
            turnCount = 0;

            RefreshAvailableActions();

            Debug.Log($"Game started: {cartridge.Spec.title} " +
                      $"({(cartridge.IsCreature ? "creature" : "mystery")}, " +
                      $"{cartridge.NumDimensions} dims)");
        }

        /// <summary>
        /// Called when the player taps an action card.
        /// </summary>
        public void OnPlayerAction(string tokenId)
        {
            if (!gameActive || !isPlayerTurn) return;

            var token = cartridge.GetToken(tokenId);
            if (token == null) return;

            // Place player token
            tokenManager.PlaceToken(token);
            string expr = cartridge.GetExpression(tokenId);
            Debug.Log($"YOU: {tokenId} — {expr}");
            // narrativeOverlay?.Show(expr);

            isPlayerTurn = false;
            turnCount++;

            // Creature: periodic token reset
            if (cartridge.IsCreature && turnCount % creatureResetInterval == 0)
                tokenManager.ResetPlacedForCreature();

            // Engine responds with a scene
            StartCoroutine(EngineSceneTurn());
        }

        /// <summary>
        /// Engine produces N tokens (one per dimension) and plays the scene.
        /// </summary>
        private IEnumerator EngineSceneTurn()
        {
            // TODO: Replace with actual ONNX inference
            // var sceneTokens = modelRuntime.PredictScene(tokenManager.GetContextTokenIds());
            // For now, simulate with random engine tokens
            var sceneTokens = SimulateScenePrediction();

            // Play each scene token with stagger
            string[] tokenIds = new string[sceneTokens.Count];
            for (int i = 0; i < sceneTokens.Count; i++)
            {
                var tok = sceneTokens[i];
                tokenManager.PlaceToken(tok);
                tokenIds[i] = tok.id;

                string expr = cartridge.GetExpression(tok.id);
                Debug.Log($"  FIELD[d{i}]: {tok.id} — {expr}");
            }

            // Play the visual scene
            if (animationRouter != null)
                animationRouter.PlayScene(tokenIds, sceneStaggerDelay);

            // Apply creature decay
            tokenManager.ApplyDecay(turnDecay);

            // Wait for scene to play out
            yield return new WaitForSeconds(sceneTokens.Count * sceneStaggerDelay + 1f);

            // Check game state
            float conv = tokenManager.ConvergenceScore;
            Debug.Log($"  Convergence: {conv:P0} ({string.Join(", ", FormatDimensions())})");

            if (!cartridge.IsCreature && tokenManager.IsConverged())
            {
                Debug.Log("=== CASE SOLVED ===");
                gameActive = false;
                yield break;
            }

            // Next player turn
            isPlayerTurn = true;
            RefreshAvailableActions();
        }

        /// <summary>
        /// Update the available action cards for the player.
        /// </summary>
        private void RefreshAvailableActions()
        {
            var available = tokenManager.GetAvailablePlayerTokens();
            Debug.Log($"Available actions: {available.Count}");

            // TODO: Update ActionBar UI
            // actionBar?.SetActions(available);

            // For debugging: list first 5
            for (int i = 0; i < Mathf.Min(5, available.Count); i++)
                Debug.Log($"  [{i}] {available[i].id}");
        }

        /// <summary>
        /// Temporary: simulate scene prediction until ONNX runtime is wired up.
        /// Returns one random engine token per dimension.
        /// </summary>
        private List<TokenDef> SimulateScenePrediction()
        {
            var engineTokens = cartridge.GetEngineTokens();
            var scene = new List<TokenDef>();

            for (int d = 0; d < cartridge.NumDimensions; d++)
            {
                // Pick a random engine token with weight on this dimension
                var candidates = new List<TokenDef>();
                foreach (var tok in engineTokens)
                {
                    if (d < tok.attractor_weights.Length &&
                        Mathf.Abs(tok.attractor_weights[d]) > 0.05f)
                        candidates.Add(tok);
                }

                if (candidates.Count > 0)
                    scene.Add(candidates[Random.Range(0, candidates.Count)]);
            }

            return scene;
        }

        private string[] FormatDimensions()
        {
            var dims = tokenManager.DimensionValues;
            var result = new string[dims.Length];
            for (int i = 0; i < dims.Length; i++)
                result[i] = $"d{i}={dims[i]:F2}";
            return result;
        }

        // Need this for Path.Combine to work
        private static class Path
        {
            public static string Combine(string a, string b) =>
                System.IO.Path.Combine(a, b);
        }
    }
}
