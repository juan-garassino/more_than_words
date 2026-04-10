using System.Collections.Generic;
using UnityEngine;

namespace LivingTales.Game
{
    /// <summary>
    /// Maps Living Tales token IDs to Unity animation triggers, camera moves,
    /// weather changes, NPC spawns, and other visual effects.
    ///
    /// The router is the bridge between the symbolic token engine and the
    /// visual game world. Each token prefix maps to a set of effects.
    /// </summary>
    [System.Serializable]
    public class SceneEffect
    {
        public enum EffectType
        {
            CreatureAnimation,  // Play animation clip on creature
            CameraMove,         // Move camera to target position
            WeatherChange,      // Change weather/lighting
            SpawnNPC,           // Spawn a character at location
            DespawnNPC,         // Remove a character
            SoundFX,            // Play sound effect
            ObjectHighlight,    // Highlight an interactive object
            UIFlash,            // Flash a UI element (dimension ring)
            LightShift,         // Change lighting color/intensity
            ParticleEffect,     // Spawn particle system
        }

        public EffectType type;
        public string parameter;    // animation name, target position, sound clip, etc.
        public float delay;         // seconds to wait before playing
        public float duration;      // how long the effect lasts (0 = auto)
    }

    public class AnimationRouter : MonoBehaviour
    {
        [Header("References")]
        public Animator creatureAnimator;
        public Transform cameraRig;
        public GameObject weatherSystem;

        // Token prefix → effects mapping
        private Dictionary<string, List<SceneEffect>> effectMap = new();

        void Start()
        {
            BuildDefaultMappings();
        }

        /// <summary>
        /// Play all effects for a given token ID.
        /// </summary>
        public void PlayToken(string tokenId, float baseDelay = 0f)
        {
            string prefix = tokenId.Contains(":") ? tokenId.Split(':')[0] : tokenId;
            string name = tokenId.Contains(":") ? tokenId.Split(':')[1] : tokenId;

            // Try exact match first, then prefix match
            List<SceneEffect> effects = null;
            if (!effectMap.TryGetValue(tokenId, out effects))
                effectMap.TryGetValue(prefix, out effects);

            if (effects == null)
            {
                // Fallback: try to trigger animation by name
                if (creatureAnimator != null)
                    creatureAnimator.SetTrigger(SanitizeAnimName(name));
                return;
            }

            foreach (var effect in effects)
            {
                float delay = baseDelay + effect.delay;
                StartCoroutine(PlayEffectDelayed(effect, delay));
            }
        }

        /// <summary>
        /// Play a scene — multiple tokens with staggered timing.
        /// </summary>
        public void PlayScene(string[] tokenIds, float staggerDelay = 0.3f)
        {
            for (int i = 0; i < tokenIds.Length; i++)
            {
                if (string.IsNullOrEmpty(tokenIds[i])) continue;
                PlayToken(tokenIds[i], i * staggerDelay);
            }
        }

        private System.Collections.IEnumerator PlayEffectDelayed(SceneEffect effect, float delay)
        {
            if (delay > 0) yield return new WaitForSeconds(delay);

            switch (effect.type)
            {
                case SceneEffect.EffectType.CreatureAnimation:
                    if (creatureAnimator != null)
                        creatureAnimator.SetTrigger(effect.parameter);
                    break;

                case SceneEffect.EffectType.CameraMove:
                    // TODO: smooth camera transition to named position
                    Debug.Log($"Camera → {effect.parameter}");
                    break;

                case SceneEffect.EffectType.WeatherChange:
                    // TODO: change weather system state
                    Debug.Log($"Weather → {effect.parameter}");
                    break;

                case SceneEffect.EffectType.SpawnNPC:
                    Debug.Log($"Spawn NPC: {effect.parameter}");
                    break;

                case SceneEffect.EffectType.SoundFX:
                    // TODO: play audio clip
                    Debug.Log($"SFX: {effect.parameter}");
                    break;

                case SceneEffect.EffectType.ObjectHighlight:
                    Debug.Log($"Highlight: {effect.parameter}");
                    break;

                case SceneEffect.EffectType.UIFlash:
                    Debug.Log($"UI Flash: {effect.parameter}");
                    break;

                case SceneEffect.EffectType.LightShift:
                    Debug.Log($"Light → {effect.parameter}");
                    break;

                case SceneEffect.EffectType.ParticleEffect:
                    Debug.Log($"Particles: {effect.parameter}");
                    break;
            }
        }

        /// <summary>
        /// Build default token-to-effect mappings for creature game.
        /// </summary>
        private void BuildDefaultMappings()
        {
            // Mood tokens → creature animations
            MapPrefix("mood", SceneEffect.EffectType.CreatureAnimation, "mood_change");

            // Specific mood mappings
            MapToken("mood:happy_wag", SceneEffect.EffectType.CreatureAnimation, "wag_tail");
            MapToken("mood:low_whine", SceneEffect.EffectType.CreatureAnimation, "ears_droop");
            MapToken("mood:calm_sigh", SceneEffect.EffectType.CreatureAnimation, "lean_against");
            MapToken("mood:eager_bounce", SceneEffect.EffectType.CreatureAnimation, "play_bounce");
            MapToken("mood:curious_sniff", SceneEffect.EffectType.CreatureAnimation, "look_at_door");

            // Need/decay tokens
            MapToken("need:belly_rumble", SceneEffect.EffectType.CreatureAnimation, "look_at_bowl");
            MapToken("decay:belly_rumble", SceneEffect.EffectType.CreatureAnimation, "look_at_bowl");
            MapToken("decay:skipped_meal", SceneEffect.EffectType.CreatureAnimation, "ears_droop");
            MapToken("decay:sick_shiver", SceneEffect.EffectType.CreatureAnimation, "shiver");
            MapToken("decay:caught_chill", SceneEffect.EffectType.CreatureAnimation, "shiver");

            // Recovery tokens
            MapToken("recovery:meal_served", SceneEffect.EffectType.CreatureAnimation, "eat_bowl");
            MapToken("recovery:forgiven_wag", SceneEffect.EffectType.CreatureAnimation, "wag_tail");
            MapToken("recovery:warm_lean", SceneEffect.EffectType.CreatureAnimation, "lean_against");

            // Mischief
            MapPrefix("mischief", SceneEffect.EffectType.CreatureAnimation, "startle");

            // Location tokens → camera moves
            MapPrefix("location", SceneEffect.EffectType.CameraMove, "location_change");
            MapToken("location:cozy_den", SceneEffect.EffectType.CameraMove, "den_view");
            MapToken("location:kitchen_corner", SceneEffect.EffectType.CameraMove, "kitchen_view");
            MapToken("location:sunny_garden", SceneEffect.EffectType.CameraMove, "garden_view");
            MapToken("location:front_step", SceneEffect.EffectType.CameraMove, "porch_view");

            // Context tokens → weather/environment
            MapToken("context:warm_sun", SceneEffect.EffectType.WeatherChange, "sunny");
            MapToken("context:rain_outside", SceneEffect.EffectType.WeatherChange, "rain");
            MapToken("context:snow_falling", SceneEffect.EffectType.WeatherChange, "snow");
            MapToken("context:cold_draft", SceneEffect.EffectType.WeatherChange, "cold");
            MapToken("context:morning_dew", SceneEffect.EffectType.LightShift, "morning");
            MapToken("context:birdsong", SceneEffect.EffectType.SoundFX, "birds");
            MapToken("context:thunder_rumble", SceneEffect.EffectType.SoundFX, "thunder");

            // Companion tokens → NPC spawns
            MapPrefix("companion", SceneEffect.EffectType.SpawnNPC, "visitor");
            MapToken("companion:mail_carrier", SceneEffect.EffectType.SpawnNPC, "mail_carrier");
            MapToken("companion:vet_nurse", SceneEffect.EffectType.SpawnNPC, "vet_nurse");
            MapToken("companion:hedgehog_visitor", SceneEffect.EffectType.SpawnNPC, "hedgehog");

            // Action tokens → creature response animations
            MapToken("action:fill_bowl", SceneEffect.EffectType.CreatureAnimation, "eat_bowl");
            MapToken("action:toss_ball", SceneEffect.EffectType.CreatureAnimation, "play_bounce");
            MapToken("action:scratch_chin", SceneEffect.EffectType.CreatureAnimation, "lean_against");
            MapToken("action:dim_lamp", SceneEffect.EffectType.LightShift, "dim");
            MapToken("action:wrap_blanket", SceneEffect.EffectType.CreatureAnimation, "sleep_curl");
        }

        private void MapPrefix(string prefix, SceneEffect.EffectType type, string param)
        {
            if (!effectMap.ContainsKey(prefix))
                effectMap[prefix] = new List<SceneEffect>();
            effectMap[prefix].Add(new SceneEffect { type = type, parameter = param });
        }

        private void MapToken(string tokenId, SceneEffect.EffectType type, string param, float delay = 0f)
        {
            if (!effectMap.ContainsKey(tokenId))
                effectMap[tokenId] = new List<SceneEffect>();
            effectMap[tokenId].Add(new SceneEffect { type = type, parameter = param, delay = delay });
        }

        private string SanitizeAnimName(string name)
        {
            return name.Replace(" ", "_").ToLower();
        }
    }
}
