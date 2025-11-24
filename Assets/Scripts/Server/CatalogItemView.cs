using System;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class CatalogItemView : MonoBehaviour
{
    [Header("UI References")]
    [SerializeField] private TMP_Text nameLabel;// ★ TMP 사용 가능하게 수정
    [SerializeField] private TextMeshProUGUI priceLabel;
    [SerializeField] private Toggle equipToggle;
    [SerializeField] private Button purchaseButton;
    [SerializeField] private GameObject ownedIndicator;

    private CatalogItemData item;
    private Action<CatalogItemData, bool> onToggleChanged;
    private Func<CatalogItemData, Task> onPurchaseRequested;
    private bool suppressToggleEvent;

    public CatalogItemData Item => item;

    public void Initialize(CatalogItemData itemData, Action<CatalogItemData, bool> toggleCallback, Func<CatalogItemData, Task> purchaseCallback)
    {
        item = itemData;
        onToggleChanged = toggleCallback;
        onPurchaseRequested = purchaseCallback;

        if (nameLabel != null)
        {
            nameLabel.text = itemData.Name;
        }

        if (priceLabel != null)
        {
            priceLabel.text = itemData.Price > 0 ? itemData.Price.ToString() : "Free";
        }

        if (equipToggle != null)
        {
            equipToggle.onValueChanged.AddListener(OnToggleValueChanged);
        }

        if (purchaseButton != null)
        {
            purchaseButton.onClick.AddListener(OnPurchaseButtonClicked);
        }
    }

    public void SetToggleGroup(ToggleGroup group)
    {
        if (equipToggle != null)
        {
            equipToggle.group = group;
        }
    }

    public void SetOwned(bool owned)
    {
        if (ownedIndicator != null)
        {
            ownedIndicator.SetActive(owned);
        }

        if (purchaseButton != null)
        {
            purchaseButton.gameObject.SetActive(!owned && item != null && item.Price > 0);
        }
    }

    public void SetSelectedWithoutNotify(bool isSelected)
    {
        if (equipToggle == null)
        {
            return;
        }

        suppressToggleEvent = true;
        equipToggle.isOn = isSelected;
        suppressToggleEvent = false;
    }

    public void SetPurchaseInteractable(bool canPurchase)
    {
        if (purchaseButton != null)
        {
            purchaseButton.interactable = canPurchase;
        }
    }

    private void OnToggleValueChanged(bool isOn)
    {
        if (suppressToggleEvent)
        {
            return;
        }

        onToggleChanged?.Invoke(item, isOn);
    }

    private async void OnPurchaseButtonClicked()
    {
        if (onPurchaseRequested == null || item == null)
        {
            return;
        }

        purchaseButton.interactable = false;
        try
        {
            await onPurchaseRequested(item);
        }
        finally
        {
            purchaseButton.interactable = true;
        }
    }

    private void OnDestroy()
    {
        if (equipToggle != null)
        {
            equipToggle.onValueChanged.RemoveListener(OnToggleValueChanged);
        }

        if (purchaseButton != null)
        {
            purchaseButton.onClick.RemoveListener(OnPurchaseButtonClicked);
        }
    }
}
