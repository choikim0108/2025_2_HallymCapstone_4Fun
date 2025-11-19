using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Firebase.Firestore;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class CatalogController : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField] private CurrencyManager currencyManager;
    [SerializeField] private CharacterPreviewController previewController;

    [Header("UI References")]
    [SerializeField] private Transform tabContainer;
    [SerializeField] private GameObject tabButtonPrefab;
    [SerializeField] private Transform itemListContainer;
    [SerializeField] private GameObject itemViewPrefab;
    [SerializeField] private Button applyButton;
    [SerializeField] private Button revertButton;
    [SerializeField] private Button lobbyButton;
    [SerializeField] private GameObject loadoutSavedIndicator;
    [SerializeField] private GameObject lobbyButtonIndicator;
    [SerializeField] private float savedIndicatorDuration = 1.5f;

    [Header("Behaviour")]
    [SerializeField] private bool autoSelectFirstTab = true;

    private FirebaseFirestore db;
    private string userId;

    private readonly Dictionary<string, List<CatalogItemData>> itemsByCategory = new();
    private readonly Dictionary<string, string> previewLoadout = new();
    private readonly Dictionary<string, CatalogItemView> activeItemViews = new();

    private string activeCategory;
    private bool catalogLoaded;
    private bool catalogInitializing;
    private bool applyingServerLoadout;
    private bool hasUnsavedChanges;
    private Coroutine savedIndicatorRoutine;

    private void Awake()
    {
        ResolveCurrencyManagerReference();
        ResolvePreviewControllerReference();
    }

    private void OnEnable()
    {
        ResolveCurrencyManagerReference();
        ResolvePreviewControllerReference();

        if (currencyManager != null)
        {
            currencyManager.InventoryReady += HandleInventoryReady;
            currencyManager.CurrencyChanged += HandleCurrencyChanged;
            currencyManager.OwnedItemsChanged += HandleOwnedItemsChanged;
            currencyManager.LoadoutChanged += HandleLoadoutChanged;

            if (currencyManager.HasInventorySnapshot && !catalogLoaded)
            {
                _ = InitializeWithInventoryAsync();
            }
        }

        if (applyButton != null)
        {
            applyButton.onClick.AddListener(OnApplyButtonClicked);
        }

        if (revertButton != null)
        {
            revertButton.onClick.AddListener(OnRevertButtonClicked);
        }

        if (lobbyButton != null)
        {
            lobbyButton.onClick.AddListener(OnLobbyButtonClicked);
        }

        if (loadoutSavedIndicator != null)
        {
            loadoutSavedIndicator.SetActive(false);
        }

        if (lobbyButtonIndicator != null)
        {
            lobbyButtonIndicator.SetActive(false);
        }
    }

    private void ResolveCurrencyManagerReference()
    {
        if (CurrencyManager.Instance != null && currencyManager != CurrencyManager.Instance)
        {
            currencyManager = CurrencyManager.Instance;
        }

        if (currencyManager == null)
        {
            currencyManager = FindFirstObjectByType<CurrencyManager>();
        }
    }

    private void ResolvePreviewControllerReference()
    {
        if (previewController == null)
        {
            previewController = FindFirstObjectByType<CharacterPreviewController>();
        }
    }

    private void OnDisable()
    {
        if (currencyManager != null)
        {
            currencyManager.InventoryReady -= HandleInventoryReady;
            currencyManager.CurrencyChanged -= HandleCurrencyChanged;
            currencyManager.OwnedItemsChanged -= HandleOwnedItemsChanged;
            currencyManager.LoadoutChanged -= HandleLoadoutChanged;
        }

        if (applyButton != null)
        {
            applyButton.onClick.RemoveListener(OnApplyButtonClicked);
        }

        if (revertButton != null)
        {
            revertButton.onClick.RemoveListener(OnRevertButtonClicked);
        }

        if (lobbyButton != null)
        {
            lobbyButton.onClick.RemoveListener(OnLobbyButtonClicked);
        }

        if (loadoutSavedIndicator != null)
        {
            loadoutSavedIndicator.SetActive(false);
        }

        if (lobbyButtonIndicator != null)
        {
            lobbyButtonIndicator.SetActive(false);
        }
    }

    private void HandleInventoryReady()
    {
        if (!catalogLoaded && !catalogInitializing)
        {
            _ = InitializeWithInventoryAsync();
        }
    }

    private async Task InitializeWithInventoryAsync()
    {
        if (catalogLoaded || catalogInitializing)
        {
            return;
        }

        if (currencyManager == null)
        {
            Debug.LogError("CatalogController: CurrencyManager 의존성이 설정되지 않았습니다.");
            return;
        }

        catalogInitializing = true;

        db = currencyManager.Database;
        userId = currencyManager.UserId;

        if (db == null || string.IsNullOrEmpty(userId))
        {
            Debug.LogWarning("CatalogController: Firebase 정보가 아직 준비되지 않았습니다.");
            catalogInitializing = false;
            return;
        }

        previewLoadout.Clear();
        foreach (KeyValuePair<string, string> entry in currencyManager.GetCurrentLoadoutSnapshot())
        {
            previewLoadout[entry.Key] = entry.Value;
        }

        if (previewController != null)
        {
            previewController.ApplyLoadout(previewLoadout, true);
        }

        hasUnsavedChanges = false;
        UpdateLobbyIndicator();

        Debug.Log("CatalogController: 초기 로드아웃 적용 완료. 카탈로그 데이터를 불러옵니다.");
        await LoadCatalogAsync();
        BuildTabs();

        if (autoSelectFirstTab && itemsByCategory.Count > 0)
        {
            string firstCategory = itemsByCategory.Keys.OrderBy(category => category).First();
            HandleTabSelection(firstCategory);
        }
        else if (!string.IsNullOrEmpty(activeCategory))
        {
            HandleTabSelection(activeCategory);
        }

        catalogLoaded = true;
        catalogInitializing = false;
    }

    private async Task LoadCatalogAsync()
    {
        if (db == null)
        {
            Debug.LogWarning("CatalogController: Firestore DB 인스턴스가 없습니다.");
            return;
        }

        Debug.Log("CatalogController: Firestore 'items' 컬렉션 조회를 시작합니다.");
        QuerySnapshot snapshot = await db.Collection("items").GetSnapshotAsync();
        int documentCount = snapshot.Documents.Count();
        Debug.Log($"CatalogController: Firestore에서 {documentCount}개의 아이템 문서를 수신했습니다.");

        itemsByCategory.Clear();

        foreach (DocumentSnapshot document in snapshot.Documents)
        {
            Dictionary<string, object> data = document.ToDictionary();

            if (!data.TryGetValue("category", out object categoryObj) || categoryObj is not string category || string.IsNullOrWhiteSpace(category))
            {
                Debug.LogWarning($"CatalogController: item {document.Id} 의 category 필드가 없습니다.");
                continue;
            }

            string name = data.TryGetValue("name", out object nameObj) && nameObj is string nameValue && !string.IsNullOrWhiteSpace(nameValue)
                ? nameValue
                : document.Id;

            int price = ParsePrice(data.TryGetValue("price", out object priceObj) ? priceObj : null);

            CatalogItemData item = new()
            {
                Id = document.Id,
                Name = name,
                Category = category,
                Price = price,
                Description = data.TryGetValue("description", out object descObj) ? descObj as string : null,
                PrefabAddress = data.TryGetValue("prefab", out object prefabObj) ? prefabObj as string : null
            };

            if (!itemsByCategory.TryGetValue(category, out List<CatalogItemData> list))
            {
                list = new List<CatalogItemData>();
                itemsByCategory[category] = list;
            }

            list.Add(item);
        }

        foreach (List<CatalogItemData> list in itemsByCategory.Values)
        {
            list.Sort((left, right) => string.CompareOrdinal(left.Name, right.Name));
        }

        if (itemsByCategory.Count == 0)
        {
            Debug.LogWarning("CatalogController: 분류된 아이템이 없습니다. 탭과 목록이 생성되지 않습니다.");
        }
        else
        {
            string summary = string.Join(", ", itemsByCategory.Select(pair => $"{pair.Key}:{pair.Value.Count}"));
            Debug.Log($"CatalogController: 카테고리 분포 - {summary}");
        }
    }

    private void BuildTabs()
    {
        if (tabContainer == null || tabButtonPrefab == null)
        {
            Debug.LogWarning("CatalogController: 탭 UI 설정이 누락되었습니다.");
            return;
        }

        foreach (Transform child in tabContainer)
        {
            Destroy(child.gameObject);
        }

        Debug.Log($"CatalogController: {itemsByCategory.Count}개의 카테고리에 대한 탭을 생성합니다.");

        ToggleGroup tabToggleGroup = tabContainer.GetComponent<ToggleGroup>();
        if (tabToggleGroup == null)
        {
            tabToggleGroup = tabContainer.gameObject.AddComponent<ToggleGroup>();
        }
        tabToggleGroup.allowSwitchOff = false;

        foreach (string category in itemsByCategory.Keys.OrderBy(cat => cat))
        {
            GameObject tabInstance = Instantiate(tabButtonPrefab, tabContainer);
            tabInstance.name = $"Tab_{category}";

            string captured = category;

            if (tabInstance.TryGetComponent(out Toggle toggle))
            {
                toggle.group = tabToggleGroup;
                toggle.onValueChanged.AddListener(isOn =>
                {
                    if (isOn)
                    {
                        HandleTabSelection(captured);
                    }
                });

                bool shouldSelect = (!string.IsNullOrEmpty(activeCategory) && activeCategory == captured) || (string.IsNullOrEmpty(activeCategory) && tabToggleGroup.AnyTogglesOn() == false);
                if (shouldSelect)
                {
                    toggle.isOn = true;
                }
            }
            else if (tabInstance.TryGetComponent(out Button button))
            {
                button.onClick.AddListener(() => HandleTabSelection(captured));
            }

            TextMeshProUGUI label = tabInstance.GetComponentInChildren<TextMeshProUGUI>();
            if (label != null)
            {
                label.text = category;
            }
        }
    }

    private void HandleTabSelection(string category)
    {
        if (string.IsNullOrWhiteSpace(category) || !itemsByCategory.ContainsKey(category))
        {
            return;
        }

        activeCategory = category;
        Debug.Log($"CatalogController: '{category}' 탭을 선택했습니다.");
        RefreshItemList();
    }

    private void RefreshItemList()
    {
        if (itemListContainer == null || itemViewPrefab == null)
        {
            Debug.LogWarning("CatalogController: 아이템 리스트 UI 설정이 누락되었습니다.");
            return;
        }

        foreach (Transform child in itemListContainer)
        {
            Destroy(child.gameObject);
        }

        activeItemViews.Clear();

        if (string.IsNullOrEmpty(activeCategory) || !itemsByCategory.TryGetValue(activeCategory, out List<CatalogItemData> items))
        {
            if (catalogLoaded)
            {
                Debug.LogWarning("CatalogController: 활성 카테고리가 없거나 아이템을 찾지 못했습니다.");
            }
            return;
        }

        ToggleGroup toggleGroup = itemListContainer.GetComponent<ToggleGroup>();
        if (toggleGroup == null)
        {
            toggleGroup = itemListContainer.gameObject.AddComponent<ToggleGroup>();
        }
        toggleGroup.allowSwitchOff = true;

        Debug.Log($"CatalogController: '{activeCategory}' 카테고리에서 {items.Count}개의 아이템 셀을 생성합니다.");
        foreach (CatalogItemData item in items)
        {
            GameObject viewObject = Instantiate(itemViewPrefab, itemListContainer);
            viewObject.name = $"Item_{item.Id}";

            CatalogItemView view = viewObject.GetComponent<CatalogItemView>();
            if (view == null)
            {
                Debug.LogWarning("CatalogController: itemViewPrefab 에 CatalogItemView 컴포넌트가 필요합니다.");
                continue;
            }

            view.Initialize(item, OnItemToggleChanged, HandlePurchaseRequestAsync);
            view.SetToggleGroup(toggleGroup);

            bool isOwned = currencyManager != null && currencyManager.IsItemOwned(item.Id);
            view.SetOwned(isOwned);

            bool isEquipped = previewLoadout.TryGetValue(item.Category, out string equippedId) && equippedId == item.Id;
            view.SetSelectedWithoutNotify(isEquipped);

            bool canPurchase = isOwned || item.Price <= 0 || (currencyManager != null && currencyManager.CurrentCurrency >= item.Price);
            view.SetPurchaseInteractable(canPurchase);

            activeItemViews[item.Id] = view;
        }
    }

    private void OnItemToggleChanged(CatalogItemData item, bool isOn)
    {
        if (item == null || applyingServerLoadout)
        {
            return;
        }

        if (!isOn)
        {
            if (previewLoadout.Remove(item.Category) && previewController != null)
            {
                previewController.ApplyPreviewItem(item.Category, null);
            }

            MarkUnsavedChanges();
            return;
        }

        previewLoadout[item.Category] = item.Id;

        if (previewController != null)
        {
            previewController.ApplyPreviewItem(item.Category, item.Id);
        }

        MarkUnsavedChanges();
    }

    private async Task HandlePurchaseRequestAsync(CatalogItemData item)
    {
        if (item == null || currencyManager == null)
        {
            return;
        }

        CurrencyManager.PurchaseResult result = await currencyManager.PurchaseItemAsync(item);

        switch (result)
        {
            case CurrencyManager.PurchaseResult.Success:
                Debug.Log($"CatalogController: '{item.Name}' 구매 완료");
                break;
            case CurrencyManager.PurchaseResult.InsufficientFunds:
                Debug.LogWarning("CatalogController: 재화가 부족합니다.");
                break;
            case CurrencyManager.PurchaseResult.AlreadyOwned:
                Debug.Log("CatalogController: 이미 소유한 아이템입니다.");
                break;
            case CurrencyManager.PurchaseResult.NotInitialized:
                Debug.LogWarning("CatalogController: Firebase 초기화가 완료되지 않았습니다.");
                break;
            default:
                Debug.LogWarning("CatalogController: 아이템 구매에 실패했습니다.");
                break;
        }
    }

    private void HandleCurrencyChanged(long newCurrency)
    {
        foreach (KeyValuePair<string, CatalogItemView> pair in activeItemViews)
        {
            CatalogItemData item = pair.Value.Item;
            if (item == null)
            {
                continue;
            }

            bool canPurchase = currencyManager != null && (currencyManager.IsItemOwned(item.Id) || item.Price <= 0 || newCurrency >= item.Price);
            pair.Value.SetPurchaseInteractable(canPurchase);
        }
    }

    private void HandleOwnedItemsChanged(IReadOnlyCollection<string> owned)
    {
        foreach (KeyValuePair<string, CatalogItemView> pair in activeItemViews)
        {
            CatalogItemData item = pair.Value.Item;
            if (item == null)
            {
                continue;
            }

            bool isOwned = currencyManager != null && currencyManager.IsItemOwned(item.Id);
            pair.Value.SetOwned(isOwned);

            bool canPurchase = isOwned || item.Price <= 0 || (currencyManager != null && currencyManager.CurrentCurrency >= item.Price);
            pair.Value.SetPurchaseInteractable(canPurchase);
        }
    }

    private void HandleLoadoutChanged(IReadOnlyDictionary<string, string> newLoadout)
    {
        applyingServerLoadout = true;

        previewLoadout.Clear();
        if (newLoadout != null)
        {
            foreach (KeyValuePair<string, string> entry in newLoadout)
            {
                previewLoadout[entry.Key] = entry.Value;
            }
        }

        if (previewController != null)
        {
            previewController.ApplyLoadout(previewLoadout, true);
        }

        RefreshItemList();
        applyingServerLoadout = false;
        hasUnsavedChanges = false;
        UpdateLobbyIndicator();
    }

    private void OnApplyButtonClicked()
    {
        if (currencyManager == null)
        {
            return;
        }

        _ = SaveLoadoutAsync();
    }

    private async Task SaveLoadoutAsync()
    {
        bool success = await currencyManager.SaveLoadoutAsync(previewLoadout);
        if (!success)
        {
            Debug.LogWarning("CatalogController: 로드아웃 저장에 실패했습니다.");
            return;
        }

        ShowSavedIndicator();
        hasUnsavedChanges = false;
        UpdateLobbyIndicator();
    }

    private void OnRevertButtonClicked()
    {
        if (currencyManager == null)
        {
            return;
        }

        HandleLoadoutChanged(currencyManager.GetCurrentLoadoutSnapshot());
    }

    private void OnLobbyButtonClicked()
    {
        if (currencyManager == null)
        {
            SceneManager.LoadScene("LobbyScene");
            return;
        }

        if (hasUnsavedChanges)
        {
            Debug.Log("CatalogController: 저장되지 않은 변경 사항을 초기 로드아웃으로 되돌립니다.");
            HandleLoadoutChanged(currencyManager.GetCurrentLoadoutSnapshot());
        }

        SceneManager.LoadScene("LobbyScene");
    }

    private static int ParsePrice(object priceObj)
    {
        if (priceObj == null)
        {
            return 0;
        }

        try
        {
            switch (priceObj)
            {
                case int value:
                    return value;
                case long valueLong:
                    return (int)Math.Max(int.MinValue, Math.Min(int.MaxValue, valueLong));
                case double valueDouble:
                    return (int)Math.Round(valueDouble);
                case float valueFloat:
                    return (int)Math.Round(valueFloat);
                case string valueString when int.TryParse(valueString, out int parsed):
                    return parsed;
                default:
                    return 0;
            }
        }
        catch
        {
            return 0;
        }
    }

    private void ShowSavedIndicator()
    {
        if (loadoutSavedIndicator == null)
        {
            return;
        }

        if (savedIndicatorRoutine != null)
        {
            StopCoroutine(savedIndicatorRoutine);
        }

        savedIndicatorRoutine = StartCoroutine(CoShowSavedIndicator());
    }

    private IEnumerator CoShowSavedIndicator()
    {
        loadoutSavedIndicator.SetActive(true);
        yield return new WaitForSeconds(savedIndicatorDuration);
        loadoutSavedIndicator.SetActive(false);
        savedIndicatorRoutine = null;
    }

    private void MarkUnsavedChanges()
    {
        hasUnsavedChanges = true;
        UpdateLobbyIndicator();
    }

    private void UpdateLobbyIndicator()
    {
        if (lobbyButtonIndicator != null)
        {
            lobbyButtonIndicator.SetActive(hasUnsavedChanges);
        }
    }
}
