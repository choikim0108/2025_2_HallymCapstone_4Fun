using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Firebase;
using Firebase.Firestore;
using Photon.Pun;
using TMPro;
using UnityEngine;
using ExitGames.Client.Photon;

public class CurrencyManager : MonoBehaviour
{
    public static CurrencyManager Instance { get; private set; }

    [Header("UI")]
    [SerializeField] private TextMeshProUGUI currencyText;

    [Header("Inventory Defaults")]
    [SerializeField] private List<string> defaultOwnedItemIds = new List<string>();

    [SerializeField] private List<CategoryDefaultItem> defaultCategoryLoadout = new List<CategoryDefaultItem>();

    private FirebaseFirestore db;
    private string userId;

    // [중요] 실시간 리스너를 저장할 변수 (씬이 파괴될 때 구독을 중지해야 함)
    private ListenerRegistration currencyListener;
    private long currentCurrency;
    private static readonly StringComparer ItemIdComparer = StringComparer.OrdinalIgnoreCase;
    private readonly HashSet<string> ownedItemIds = new HashSet<string>(ItemIdComparer);
    private readonly Dictionary<string, string> currentLoadout = new Dictionary<string, string>(ItemIdComparer);
    private bool inventorySnapshotReceived;
    private static readonly Dictionary<string, string> lastKnownLoadout = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    private static string lastSerializedLoadoutJson;

    public event Action<long> CurrencyChanged;
    public event Action<IReadOnlyCollection<string>> OwnedItemsChanged;
    public event Action<IReadOnlyDictionary<string, string>> LoadoutChanged;
    public event Action InventoryReady;

    public FirebaseFirestore Database => db;
    public string UserId => userId;
    public long CurrentCurrency => currentCurrency;
    public bool HasInventorySnapshot => inventorySnapshotReceived;
    public IReadOnlyCollection<string> OwnedItemIds => ownedItemIds;
    public IReadOnlyDictionary<string, string> CurrentLoadout => currentLoadout;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);
        NormalizeDefaultOwnership();
    }

    private void OnEnable()
    {
        if (FirebaseManager.Instance != null)
        {
            FirebaseManager.Instance.OnFirebaseInitialized += HandleFirebaseInitialized;
        }

        _ = InitializeWhenReadyAsync();
    }

    private void OnDisable()
    {
        if (FirebaseManager.Instance != null)
        {
            FirebaseManager.Instance.OnFirebaseInitialized -= HandleFirebaseInitialized;
        }
    }

    private async System.Threading.Tasks.Task InitializeWhenReadyAsync()
    {
        FirebaseManager manager = FirebaseManager.Instance;

        while (manager == null)
        {
            await System.Threading.Tasks.Task.Yield();
            manager = FirebaseManager.Instance;
        }

        manager.OnFirebaseInitialized -= HandleFirebaseInitialized;
        manager.OnFirebaseInitialized += HandleFirebaseInitialized;

        if (manager.IsFirebaseInitialized)
        {
            HandleFirebaseInitialized(true, "Initialized");
        }
    }

    private void HandleFirebaseInitialized(bool success, string message)
    {
        if (!success)
        {
            return;
        }

        FirebaseManager manager = FirebaseManager.Instance;
        if (manager == null)
        {
            return;
        }

        db = manager.DB;
        userId = manager.UserId;

        if (!string.IsNullOrEmpty(userId) && currencyListener == null)
        {
            ListenToCurrencyChanges();

            if (!inventorySnapshotReceived)
            {
                ApplyCachedStateIfAvailable();
            }
        }
    }

    /// <summary>
    /// 내 유저 문서의 'currency' 필드 변경을 실시간으로 감지합니다.
    /// </summary>
    void ListenToCurrencyChanges()
    {
        if (db == null || string.IsNullOrEmpty(userId)) return;

        // 1. 내 유저 문서의 참조를 가져옴
        DocumentReference userDocRef = db.Collection("users").Document(userId);

        // 2. Listen으로 실시간 구독 시작 (Unity Firestore SDK에서는 Listen 사용)
        currencyListener = userDocRef.Listen(snapshot =>
        {
            if (snapshot == null)
            {
                Debug.LogWarning("Currency Snapshot: null");
                return;
            }

            if (!snapshot.Exists)
            {
                Debug.Log("Currency Snapshot: 문서가 존재하지 않음");
                return;
            }

            // Firestore는 숫자를 long으로 반환하는 것이 일반적
            if (snapshot.TryGetValue("currency", out long newCurrency))
            {
                UpdateCurrency(newCurrency);
            }
            else
            {
                // currency 필드가 없다면 0으로 초기화
                UpdateCurrency(0);
            }

            if (snapshot.TryGetValue("owned_items", out List<object> ownedRaw))
            {
                SyncOwnedItems(ObjectListToStringEnumerable(ownedRaw));
            }
            else
            {
                SyncOwnedItems(null);
            }

            bool hasLoadoutField = snapshot.TryGetValue("current_loadout", out Dictionary<string, object> loadoutRaw);
            if (hasLoadoutField)
            {
                SyncLoadout(loadoutRaw, true);
            }
            else
            {
                SyncLoadout(null, false);
            }

            if (!inventorySnapshotReceived)
            {
                inventorySnapshotReceived = true;
                InventoryReady?.Invoke();
            }
        });
    }

    /// <summary>
    /// 이 오브젝트가 파괴될 때 (씬 이동 등)
    /// </summary>
    void OnDestroy()
    {
        // [중요] 리스너를 중지(구독 취소)해서 메모리 누수를 방지합니다.
        if (currencyListener != null)
        {
            currencyListener.Stop();
            currencyListener = null;
        }

        if (Instance == this)
        {
            Instance = null;
        }
    }

    public bool IsItemOwned(string itemId)
    {
        if (string.IsNullOrWhiteSpace(itemId))
        {
            return false;
        }

        return ownedItemIds.Contains(itemId) || defaultOwnedItemIds.Any(id => ItemIdComparer.Equals(id, itemId));
    }

    public bool TryGetEquippedItem(string category, out string itemId)
    {
        if (string.IsNullOrEmpty(category))
        {
            itemId = null;
            return false;
        }

        return currentLoadout.TryGetValue(category, out itemId);
    }

    public async Task EnsureInitialDataLoadedAsync(bool forceReload = false)
    {
        FirebaseManager manager = FirebaseManager.Instance;
        if (manager != null)
        {
            if (db == null)
            {
                db = manager.DB;
            }

            if (string.IsNullOrEmpty(userId))
            {
                userId = manager.UserId;
            }
        }

        if (!IsReady())
        {
            return;
        }

        if (currencyListener == null)
        {
            ListenToCurrencyChanges();
        }

        if (inventorySnapshotReceived && !forceReload)
        {
            return;
        }

        Dictionary<string, string> loadout = await CurrencyManagerLoader.FetchLoadoutAsync(db, userId);
        if (loadout != null && loadout.Count > 0)
        {
            SyncLoadout(loadout.ToDictionary(pair => pair.Key, pair => (object)pair.Value), true);
        }
    }

    public Dictionary<string, string> GetCurrentLoadoutSnapshot()
    {
        return currentLoadout.ToDictionary(pair => pair.Key, pair => pair.Value);
    }

    public async Task<PurchaseResult> PurchaseItemAsync(CatalogItemData itemData)
    {
        if (itemData == null || string.IsNullOrWhiteSpace(itemData.Id))
        {
            Debug.LogWarning("CurrencyManager: 잘못된 아이템 데이터로 구매를 시도했습니다.");
            return PurchaseResult.InvalidRequest;
        }

        if (!IsReady())
        {
            Debug.LogWarning("CurrencyManager: Firebase가 초기화되기 전에 구매를 시도했습니다.");
            return PurchaseResult.NotInitialized;
        }

        if (IsItemOwned(itemData.Id))
        {
            return PurchaseResult.AlreadyOwned;
        }

        try
        {
            DocumentReference userDocRef = db.Collection("users").Document(userId);

            PurchaseTransactionResult transactionResult = await db.RunTransactionAsync(async transaction =>
            {
                DocumentSnapshot snapshot = await transaction.GetSnapshotAsync(userDocRef);

                long remoteCurrency = snapshot.ContainsField("currency") ? snapshot.GetValue<long>("currency") : 0L;
                List<string> remoteOwnedList = snapshot.ContainsField("owned_items")
                    ? snapshot.GetValue<List<string>>("owned_items")
                    : new List<string>();

                if (remoteOwnedList.Contains(itemData.Id))
                {
                    return new PurchaseTransactionResult
                    {
                        Result = PurchaseResult.AlreadyOwned,
                        UpdatedCurrency = remoteCurrency,
                        OwnedItems = remoteOwnedList
                    };
                }

                if (itemData.Price > 0 && remoteCurrency < itemData.Price)
                {
                    return new PurchaseTransactionResult
                    {
                        Result = PurchaseResult.InsufficientFunds,
                        UpdatedCurrency = remoteCurrency,
                        OwnedItems = remoteOwnedList
                    };
                }

                long updatedCurrency = remoteCurrency - Math.Max(0, itemData.Price);
                if (itemData.Price <= 0 && !remoteOwnedList.Contains(itemData.Id))
                {
                    remoteOwnedList.Add(itemData.Id);
                }
                else if (itemData.Price > 0)
                {
                    remoteOwnedList.Add(itemData.Id);
                }

                transaction.Update(userDocRef, new Dictionary<string, object>
                {
                    { "currency", updatedCurrency },
                    { "owned_items", remoteOwnedList }
                });

                return new PurchaseTransactionResult
                {
                    Result = PurchaseResult.Success,
                    UpdatedCurrency = updatedCurrency,
                    OwnedItems = remoteOwnedList
                };
            });

            if (transactionResult.Result == PurchaseResult.Success)
            {
                UpdateCurrency(transactionResult.UpdatedCurrency);
                SyncOwnedItems(transactionResult.OwnedItems);
            }

            return transactionResult.Result;
        }
        catch (Exception ex)
        {
            Debug.LogError($"CurrencyManager: 구매 진행 중 오류 발생 - {ex.Message}");
            return PurchaseResult.UnknownFailure;
        }
    }

    public async Task<bool> SaveLoadoutAsync(Dictionary<string, string> desiredLoadout)
    {
        if (!IsReady())
        {
            Debug.LogWarning("CurrencyManager: Firebase 초기화 전에 로드아웃 저장을 시도했습니다.");
            return false;
        }

        if (desiredLoadout == null)
        {
            Debug.LogWarning("CurrencyManager: 로드아웃 저장 요청이 null입니다.");
            return false;
        }

        foreach (KeyValuePair<string, string> pair in desiredLoadout)
        {
            if (string.IsNullOrWhiteSpace(pair.Key) || string.IsNullOrWhiteSpace(pair.Value))
            {
                Debug.LogWarning("CurrencyManager: 로드아웃에 유효하지 않은 항목이 포함되어 있습니다.");
                return false;
            }

            string categoryKey = pair.Key.Trim();
            string itemId = pair.Value.Trim();

            if (!IsItemOwned(itemId) &&
                !IsDefaultCategoryItem(categoryKey, itemId) &&
                !IsCurrentLoadoutItem(categoryKey, itemId))
            {
                Debug.LogWarning($"CurrencyManager: 소유하지 않은 아이템({itemId})은 로드아웃에 저장할 수 없습니다.");
                return false;
            }
        }

        try
        {
            Dictionary<string, string> normalized = desiredLoadout
                .Where(pair => !string.IsNullOrWhiteSpace(pair.Key) && !string.IsNullOrWhiteSpace(pair.Value))
                .ToDictionary(pair => pair.Key.Trim(), pair => pair.Value.Trim(), StringComparer.OrdinalIgnoreCase);

            Dictionary<string, object> payload = normalized.ToDictionary(pair => pair.Key, pair => (object)pair.Value, StringComparer.OrdinalIgnoreCase);

            DocumentReference userDocRef = db.Collection("users").Document(userId);

            // UpdateAsync는 지정한 필드를 새로운 Map으로 교체하므로 기존 항목이 제거됩니다.
            await userDocRef.UpdateAsync(new Dictionary<string, object>
            {
                { "current_loadout", payload }
            });

            SyncLoadout(payload, true);
            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"CurrencyManager: 로드아웃 저장 중 오류가 발생했습니다. {ex.Message}");
            return false;
        }
    }

    private void NormalizeDefaultOwnership()
    {
        if (defaultOwnedItemIds == null)
        {
            defaultOwnedItemIds = new List<string>();
        }

        if (defaultCategoryLoadout == null)
        {
            defaultCategoryLoadout = new List<CategoryDefaultItem>();
        }

        foreach (CategoryDefaultItem entry in defaultCategoryLoadout)
        {
            if (string.IsNullOrWhiteSpace(entry.defaultItemId))
            {
                continue;
            }

            if (!defaultOwnedItemIds.Any(id => ItemIdComparer.Equals(id, entry.defaultItemId)))
            {
                defaultOwnedItemIds.Add(entry.defaultItemId);
            }
        }

        defaultOwnedItemIds = defaultOwnedItemIds
            .Where(id => !string.IsNullOrWhiteSpace(id))
            .Select(id => id.Trim())
            .Distinct(ItemIdComparer)
            .ToList();
    }

    private void UpdateCurrency(long newCurrency)
    {
        bool hasChanged = currentCurrency != newCurrency;
        currentCurrency = newCurrency;

        ApplyCurrencyToText();

        if (hasChanged)
        {
            CurrencyChanged?.Invoke(newCurrency);
            Debug.Log("실시간 재화 업데이트: " + newCurrency);
        }
    }

    private void SyncOwnedItems(IEnumerable<string> source)
    {
        HashSet<string> updated = new HashSet<string>(defaultOwnedItemIds, ItemIdComparer);

        if (source != null)
        {
            foreach (string entry in source)
            {
                if (!string.IsNullOrWhiteSpace(entry))
                {
                    updated.Add(entry.Trim());
                }
            }
        }

        if (!updated.SetEquals(ownedItemIds))
        {
            ownedItemIds.Clear();
            ownedItemIds.UnionWith(updated);
            OwnedItemsChanged?.Invoke(ownedItemIds);
        }
    }

    private void SyncOwnedItems(List<string> source)
    {
        SyncOwnedItems((IEnumerable<string>)source);
    }

    private void SyncLoadout(IDictionary<string, object> rawLoadout, bool fieldExists)
    {
        Dictionary<string, string> updated = new Dictionary<string, string>(ItemIdComparer);

        if (rawLoadout != null)
        {
            foreach (KeyValuePair<string, object> pair in rawLoadout)
            {
                if (!string.IsNullOrWhiteSpace(pair.Key) && pair.Value is string itemId && !string.IsNullOrWhiteSpace(itemId))
                {
                    string categoryKey = pair.Key.Trim();
                    string normalizedItemId = itemId.Trim();

                    if (!string.IsNullOrEmpty(categoryKey) && !string.IsNullOrEmpty(normalizedItemId))
                    {
                        updated[categoryKey] = normalizedItemId;
                    }
                }
            }
        }

        if (!fieldExists && defaultCategoryLoadout.Count > 0)
        {
            foreach (CategoryDefaultItem entry in defaultCategoryLoadout)
            {
                string categoryKey = entry.category?.Trim();
                string defaultItemId = entry.defaultItemId?.Trim();

                if (!string.IsNullOrEmpty(categoryKey) && !string.IsNullOrEmpty(defaultItemId))
                {
                    updated[categoryKey] = defaultItemId;
                }
            }
        }

        if (!DictionaryEquals(currentLoadout, updated))
        {
            currentLoadout.Clear();
            foreach (KeyValuePair<string, string> pair in updated)
            {
                currentLoadout[pair.Key] = pair.Value;
            }

            LoadoutChanged?.Invoke(new Dictionary<string, string>(currentLoadout));
            UpdateLoadoutCaches();
            BroadcastLoadoutToPhoton();
        }
        else
        {
            UpdateLoadoutCaches();
        }
    }

    private void ApplyCachedStateIfAvailable()
    {
        if (lastKnownLoadout.Count > 0 && currentLoadout.Count == 0)
        {
            foreach (KeyValuePair<string, string> entry in lastKnownLoadout)
            {
                currentLoadout[entry.Key] = entry.Value;
            }

            LoadoutChanged?.Invoke(new Dictionary<string, string>(currentLoadout));
            BroadcastLoadoutToPhoton();
        }
    }

    private static bool DictionaryEquals(Dictionary<string, string> left, Dictionary<string, string> right)
    {
        if (left.Count != right.Count)
        {
            return false;
        }

        foreach (KeyValuePair<string, string> pair in left)
        {
            if (!right.TryGetValue(pair.Key, out string otherValue) || !ItemIdComparer.Equals(otherValue, pair.Value))
            {
                return false;
            }
        }

        return true;
    }

    private static IEnumerable<string> ObjectListToStringEnumerable(IEnumerable<object> source)
    {
        if (source == null)
        {
            yield break;
        }

        foreach (object item in source)
        {
            if (item is string str && !string.IsNullOrWhiteSpace(str))
            {
                yield return str.Trim();
            }
        }
    }

    private bool IsReady()
    {
        return db != null && !string.IsNullOrWhiteSpace(userId);
    }

    private void ApplyCurrencyToText()
    {
        if (currencyText == null)
        {
            return;
        }

        currencyText.text = $"Cash: {currentCurrency}";
    }

    private bool IsDefaultCategoryItem(string category, string itemId)
    {
        if (string.IsNullOrWhiteSpace(category) || string.IsNullOrWhiteSpace(itemId))
        {
            return false;
        }

        string normalizedCategory = category.Trim();
        string normalizedItem = itemId.Trim();

        foreach (CategoryDefaultItem entry in defaultCategoryLoadout)
        {
            if (string.IsNullOrWhiteSpace(entry.category) || string.IsNullOrWhiteSpace(entry.defaultItemId))
            {
                continue;
            }

            if (string.Equals(entry.category.Trim(), normalizedCategory, StringComparison.OrdinalIgnoreCase) &&
                ItemIdComparer.Equals(entry.defaultItemId.Trim(), normalizedItem))
            {
                return true;
            }
        }

        return false;
    }

    private bool IsCurrentLoadoutItem(string category, string itemId)
    {
        if (string.IsNullOrWhiteSpace(category) || string.IsNullOrWhiteSpace(itemId))
        {
            return false;
        }

        return currentLoadout.TryGetValue(category.Trim(), out string equipped) && ItemIdComparer.Equals(equipped, itemId.Trim());
    }

    public void RegisterCurrencyText(TextMeshProUGUI text)
    {
        if (text == null)
        {
            return;
        }

        currencyText = text;
        ApplyCurrencyToText();
    }

    public void UnregisterCurrencyText(TextMeshProUGUI text)
    {
        if (currencyText == text)
        {
            currencyText = null;
        }
    }

    private void BroadcastLoadoutToPhoton()
    {
        lastSerializedLoadoutJson = LoadoutSerializer.Serialize(currentLoadout);

        TryPushCachedLoadoutToPhoton();
    }

    public static void TryPushCachedLoadoutToPhoton()
    {
        if (!PhotonNetwork.IsConnected || PhotonNetwork.LocalPlayer == null || string.IsNullOrEmpty(lastSerializedLoadoutJson))
        {
            return;
        }

        Hashtable props = new Hashtable
        {
            { "loadout", lastSerializedLoadoutJson }
        };

        PhotonNetwork.LocalPlayer.SetCustomProperties(props);
    }
    
    private void UpdateLoadoutCaches()
    {
        lastKnownLoadout.Clear();
        foreach (KeyValuePair<string, string> pair in currentLoadout)
        {
            lastKnownLoadout[pair.Key] = pair.Value;
        }

        if (lastKnownLoadout.Count > 0)
        {
            Debug.Log("CurrencyManager: Updated loadout cache -> " + string.Join(", ", lastKnownLoadout));
        }
        else
        {
            Debug.Log("CurrencyManager: Loadout cache cleared or empty.");
        }
    }

    public static Dictionary<string, string> GetLastKnownLoadoutSnapshot()
    {
        return lastKnownLoadout.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.OrdinalIgnoreCase);
    }

    [Serializable]
    public struct CategoryDefaultItem
    {
        public string category;
        public string defaultItemId;
    }

    private struct PurchaseTransactionResult
    {
        public PurchaseResult Result;
        public long UpdatedCurrency;
        public List<string> OwnedItems;
    }

    public enum PurchaseResult
    {
        Success,
        AlreadyOwned,
        InsufficientFunds,
        NotInitialized,
        InvalidRequest,
        UnknownFailure
    }
}