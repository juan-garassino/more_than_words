using UnityEngine;
using UnityEngine.UI;
using LivingTales.Engine;

namespace LivingTales.UI
{
    /// <summary>
    /// Displays creature dimension values as subtle radial rings.
    /// Each dimension is a thin arc that fills/empties based on the value.
    /// Positioned in a corner, minimal and non-intrusive.
    ///
    /// For mysteries: shows convergence progress as a single bar.
    /// </summary>
    public class DimensionRings : MonoBehaviour
    {
        [Header("References")]
        public RectTransform ringContainer;
        public Image convergenceBar;   // for mysteries: simple progress bar

        [Header("Ring Settings")]
        public float ringRadius = 40f;
        public float ringWidth = 4f;
        public float ringSpacing = 8f;

        [Header("Colors")]
        public Color healthyColor = new Color(0.5f, 0.8f, 0.4f);     // green
        public Color warningColor = new Color(0.9f, 0.8f, 0.3f);     // yellow
        public Color criticalColor = new Color(0.9f, 0.3f, 0.3f);    // red
        public Color backgroundRingColor = new Color(0.85f, 0.82f, 0.78f, 0.3f);

        // Dimension labels for creature M (7 dims)
        private static readonly string[] creatureDimLabels = {
            "Hunger", "Happiness", "Energy", "Social",
            "Comfort", "Health", "Environment"
        };

        private Image[] ringFills;
        private Image[] ringBackgrounds;
        private int numDimensions;
        private bool isCreature;

        /// <summary>
        /// Initialize rings for the loaded cartridge.
        /// </summary>
        public void Initialize(Cartridge cartridge)
        {
            numDimensions = cartridge.NumDimensions;
            isCreature = cartridge.IsCreature;

            if (isCreature)
                CreateRings();
            else
                SetupConvergenceBar();
        }

        /// <summary>
        /// Update ring fills based on current dimension values.
        /// </summary>
        public void UpdateValues(float[] dimensionValues, float convergenceScore)
        {
            if (isCreature)
            {
                for (int d = 0; d < numDimensions && d < dimensionValues.Length; d++)
                {
                    if (ringFills != null && d < ringFills.Length && ringFills[d] != null)
                    {
                        float val = dimensionValues[d];
                        ringFills[d].fillAmount = val;
                        ringFills[d].color = GetDimensionColor(val);
                    }
                }
            }
            else
            {
                // Mystery: update convergence bar
                if (convergenceBar != null)
                {
                    convergenceBar.fillAmount = convergenceScore;
                    convergenceBar.color = Color.Lerp(warningColor, healthyColor, convergenceScore);
                }
            }
        }

        /// <summary>
        /// Pulse a specific dimension ring (when it changes significantly).
        /// </summary>
        public void PulseDimension(int dimIndex)
        {
            if (dimIndex < 0 || ringFills == null || dimIndex >= ringFills.Length) return;
            // TODO: animate scale pulse on the ring
            Debug.Log($"Pulse dimension {dimIndex}");
        }

        private void CreateRings()
        {
            if (ringContainer == null) return;

            ringFills = new Image[numDimensions];
            ringBackgrounds = new Image[numDimensions];

            for (int d = 0; d < numDimensions; d++)
            {
                float radius = ringRadius + d * ringSpacing;

                // Background ring (full circle, dim)
                var bgGO = CreateRingImage($"RingBG_{d}", radius, backgroundRingColor, 1f);
                ringBackgrounds[d] = bgGO.GetComponent<Image>();

                // Fill ring (partial circle, colored)
                var fillGO = CreateRingImage($"RingFill_{d}", radius, healthyColor, 0.5f);
                ringFills[d] = fillGO.GetComponent<Image>();
            }
        }

        private GameObject CreateRingImage(string name, float radius, Color color, float fillAmount)
        {
            var go = new GameObject(name);
            go.transform.SetParent(ringContainer, false);

            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = new Vector2(radius * 2, radius * 2);
            rt.anchoredPosition = Vector2.zero;

            var image = go.AddComponent<Image>();
            image.type = Image.Type.Filled;
            image.fillMethod = Image.FillMethod.Radial360;
            image.fillOrigin = (int)Image.Origin360.Top;
            image.fillClockwise = true;
            image.fillAmount = fillAmount;
            image.color = color;
            // Note: in production, use a ring sprite; for now this is a filled circle

            return go;
        }

        private void SetupConvergenceBar()
        {
            if (convergenceBar != null)
            {
                convergenceBar.type = Image.Type.Filled;
                convergenceBar.fillMethod = Image.FillMethod.Horizontal;
                convergenceBar.fillAmount = 0f;
            }
        }

        private Color GetDimensionColor(float value)
        {
            if (value > 0.6f) return healthyColor;
            if (value > 0.3f) return warningColor;
            return criticalColor;
        }
    }
}
