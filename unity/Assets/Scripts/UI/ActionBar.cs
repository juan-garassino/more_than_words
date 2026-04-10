using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using LivingTales.Engine;

namespace LivingTales.UI
{
    /// <summary>
    /// Bottom action bar showing available player tokens as tappable cards.
    /// Scrollable horizontal list. Each card shows the token's surface expression.
    /// </summary>
    public class ActionBar : MonoBehaviour
    {
        [Header("References")]
        public Transform cardContainer;     // Horizontal layout group
        public GameObject cardPrefab;       // Card template with Text + Button

        [Header("Settings")]
        public int maxVisibleCards = 5;
        public Color cardColor = new Color(0.95f, 0.92f, 0.85f);
        public Color cardHighlightColor = new Color(1f, 0.95f, 0.80f);

        // Callback when player taps a card
        public System.Action<string> OnCardTapped;

        private List<GameObject> activeCards = new();
        private Cartridge cartridge;

        public void Initialize(Cartridge cartridge)
        {
            this.cartridge = cartridge;
        }

        /// <summary>
        /// Update the action bar with available tokens.
        /// </summary>
        public void SetActions(List<TokenDef> availableTokens)
        {
            // Clear existing cards
            foreach (var card in activeCards)
                Destroy(card);
            activeCards.Clear();

            // Shuffle to add variety
            var shuffled = new List<TokenDef>(availableTokens);
            for (int i = shuffled.Count - 1; i > 0; i--)
            {
                int j = Random.Range(0, i + 1);
                (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
            }

            // Show up to maxVisibleCards
            int count = Mathf.Min(maxVisibleCards, shuffled.Count);
            for (int i = 0; i < count; i++)
            {
                var token = shuffled[i];
                CreateCard(token);
            }
        }

        private void CreateCard(TokenDef token)
        {
            if (cardPrefab == null || cardContainer == null) return;

            var card = Instantiate(cardPrefab, cardContainer);
            card.name = $"Card_{token.id}";
            activeCards.Add(card);

            // Set card text
            var text = card.GetComponentInChildren<TextMeshProUGUI>();
            if (text != null)
            {
                // Show short name from token ID
                string name = token.id.Contains(":") ? token.id.Split(':')[1] : token.id;
                text.text = FormatTokenName(name);
                text.fontSize = 14;
            }

            // Set card color
            var image = card.GetComponent<Image>();
            if (image != null)
                image.color = cardColor;

            // Set button click
            var button = card.GetComponent<Button>();
            if (button != null)
            {
                string tokenId = token.id; // capture for closure
                button.onClick.AddListener(() =>
                {
                    OnCardTapped?.Invoke(tokenId);
                });
            }

            // Add hover highlight
            var trigger = card.AddComponent<UnityEngine.EventSystems.EventTrigger>();
            AddHoverEffect(trigger, card, image);
        }

        private void AddHoverEffect(UnityEngine.EventSystems.EventTrigger trigger,
                                     GameObject card, Image image)
        {
            if (image == null) return;

            var enterEntry = new UnityEngine.EventSystems.EventTrigger.Entry
            {
                eventID = UnityEngine.EventSystems.EventTriggerType.PointerEnter
            };
            enterEntry.callback.AddListener((_) => image.color = cardHighlightColor);
            trigger.triggers.Add(enterEntry);

            var exitEntry = new UnityEngine.EventSystems.EventTrigger.Entry
            {
                eventID = UnityEngine.EventSystems.EventTriggerType.PointerExit
            };
            exitEntry.callback.AddListener((_) => image.color = cardColor);
            trigger.triggers.Add(exitEntry);
        }

        private string FormatTokenName(string name)
        {
            // "fill_bowl" → "Fill Bowl"
            var words = name.Split('_');
            for (int i = 0; i < words.Length; i++)
            {
                if (words[i].Length > 0)
                    words[i] = char.ToUpper(words[i][0]) + words[i].Substring(1);
            }
            return string.Join(" ", words);
        }
    }
}
