using System.Collections;
using UnityEngine;
using TMPro;

namespace LivingTales.UI
{
    /// <summary>
    /// Displays narrative text (surface expressions) with typewriter effect.
    /// Fades in, holds, fades out. Can queue multiple lines.
    /// </summary>
    public class NarrativeOverlay : MonoBehaviour
    {
        [Header("References")]
        public TextMeshProUGUI narrativeText;
        public CanvasGroup canvasGroup;

        [Header("Timing")]
        public float typewriterSpeed = 0.03f;   // seconds per character
        public float holdDuration = 3.0f;        // seconds to display
        public float fadeInDuration = 0.5f;
        public float fadeOutDuration = 0.8f;

        [Header("Style")]
        public Color textColor = new Color(0.2f, 0.18f, 0.15f);
        public int fontSize = 18;

        private Queue expressionQueue = new();
        private bool isDisplaying = false;

        void Start()
        {
            if (canvasGroup != null)
                canvasGroup.alpha = 0f;
            if (narrativeText != null)
            {
                narrativeText.color = textColor;
                narrativeText.fontSize = fontSize;
            }
        }

        /// <summary>
        /// Show a narrative expression. Queues if one is already playing.
        /// </summary>
        public void Show(string expression)
        {
            if (string.IsNullOrWhiteSpace(expression)) return;
            expressionQueue.Enqueue(expression);

            if (!isDisplaying)
                StartCoroutine(DisplayLoop());
        }

        /// <summary>
        /// Show a scene of expressions (one per head token).
        /// Combines them into a flowing narrative.
        /// </summary>
        public void ShowScene(string[] expressions)
        {
            // Combine non-empty expressions into one or two sentences
            var lines = new System.Collections.Generic.List<string>();
            foreach (var expr in expressions)
            {
                if (!string.IsNullOrWhiteSpace(expr))
                    lines.Add(expr);
            }

            if (lines.Count == 0) return;

            // Show first 2 expressions together (the scene feeling)
            int showCount = Mathf.Min(2, lines.Count);
            string combined = string.Join("\n", lines.GetRange(0, showCount));
            Show(combined);
        }

        private IEnumerator DisplayLoop()
        {
            isDisplaying = true;

            while (expressionQueue.Count > 0)
            {
                string text = (string)expressionQueue.Dequeue();
                yield return StartCoroutine(DisplayOne(text));
            }

            isDisplaying = false;
        }

        private IEnumerator DisplayOne(string text)
        {
            if (narrativeText == null || canvasGroup == null)
                yield break;

            narrativeText.text = "";

            // Fade in
            yield return StartCoroutine(Fade(0f, 1f, fadeInDuration));

            // Typewriter effect
            for (int i = 0; i <= text.Length; i++)
            {
                narrativeText.text = text.Substring(0, i);
                yield return new WaitForSeconds(typewriterSpeed);
            }

            // Hold
            yield return new WaitForSeconds(holdDuration);

            // Fade out
            yield return StartCoroutine(Fade(1f, 0f, fadeOutDuration));

            narrativeText.text = "";
        }

        private IEnumerator Fade(float from, float to, float duration)
        {
            float elapsed = 0f;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                canvasGroup.alpha = Mathf.Lerp(from, to, elapsed / duration);
                yield return null;
            }
            canvasGroup.alpha = to;
        }

        /// <summary>
        /// Immediately clear any displayed text.
        /// </summary>
        public void Clear()
        {
            StopAllCoroutines();
            isDisplaying = false;
            expressionQueue.Clear();
            if (canvasGroup != null) canvasGroup.alpha = 0f;
            if (narrativeText != null) narrativeText.text = "";
        }
    }
}
