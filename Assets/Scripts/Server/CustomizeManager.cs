using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using CharacterCustomizationTool.Extensions;
using CharacterCustomizationTool.FaceManagement;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.ResourceManagement.AsyncOperations;

/// <summary>
/// Addressables로 내려받은 의상 프리팹에서 스킨 메시를 추출해
/// 이미 리깅된 플레이어 아바타 본의 SkinnedMeshRenderer에 적용합니다.
/// CharacterCustomizationWindow.cs의 SavePrefab()이 하던 일을 런타임으로 옮긴 형태입니다.
/// </summary>
public class CustomizeManager : MonoBehaviour
{
    [Header("Avatar root")]
    [SerializeField] private Transform characterRoot;

    [Header("Optional face helper")]
    [SerializeField] private FacePicker facePicker;

    [Header("Slot bindings (Firestore category ↔ Renderer)")]
    [SerializeField] private List<SlotBinding> slotBindings = new();

    private readonly Dictionary<string, SlotBinding> _bindingLookup = new();
    private readonly Dictionary<string, RendererSnapshot> _defaults = new();
    private readonly Dictionary<string, AsyncOperationHandle<GameObject>> _loadedHandles = new();

    private void Awake()
    {
        CacheBindings();
    }

    private void OnValidate()
    {
        if (!Application.isPlaying)
        {
            CacheBindings();
        }
    }

    private void OnDestroy()
    {
        ReleaseAllHandles();
    }

#if UNITY_EDITOR
    // 에디터에서만 테스트용 로드아웃 적용 (빌드에서는 실행되지 않음)
    private async void Start()
    {
        if (slotBindings.Count == 0) return;

        var demoLoadout = new Dictionary<string, object>
        {
            { "shirt", "default_shirt" },
            { "pants", "default_pants" },
            // "face" 테스트를 위해 "face" 카테고리와 "default_face" Addressable 키가 필요합니다.
            { "face", "default_face" } 
        };

        await ApplyLoadoutAsync(demoLoadout);
    }
#endif

    /// <summary>
    /// Firestore에서 내려받은 current_loadout 딕셔너리를 전달하면
    /// 각 카테고리별 Addressable 프리팹을 읽어 적용합니다.
    /// </summary>
    public async Task ApplyLoadoutAsync(Dictionary<string, object> loadout)
    {
        if (loadout == null)
        {
            Debug.LogWarning("CustomizeManager.ApplyLoadoutAsync: 전달된 loadout이 null 입니다.");
            return;
        }

        var normalizedLoadout = new Dictionary<string, string>();
        foreach (var entry in loadout)
        {
            var key = Normalize(entry.Key);
            if (string.IsNullOrEmpty(key)) continue;

            normalizedLoadout[key] = entry.Value?.ToString();
        }

        var equipTasks = new List<Task>();

        foreach (var kvp in normalizedLoadout)
        {
            if (!_bindingLookup.TryGetValue(kvp.Key, out var binding))
            {
                Debug.LogWarning($"CustomizeManager: '{kvp.Key}' 카테고리에 대한 슬롯 바인딩이 없습니다.");
                continue;
            }

            if (string.IsNullOrWhiteSpace(kvp.Value))
            {
                ResetSlot(kvp.Key);
                continue;
            }

            equipTasks.Add(EquipItemAsync(kvp.Key, kvp.Value, binding));
        }

        foreach (var binding in _bindingLookup)
        {
            if (!normalizedLoadout.ContainsKey(binding.Key))
            {
                ResetSlot(binding.Key);
            }
        }

        if (equipTasks.Count > 0)
        {
            await Task.WhenAll(equipTasks);
        }
    }

    private void CacheBindings()
    {
        _bindingLookup.Clear();
        _defaults.Clear();

        if (characterRoot == null && slotBindings.Any(b => b.Renderer == null))
        {
            Debug.LogWarning("CustomizeManager: characterRoot를 지정하면 하위에서 렌더러를 자동으로 찾을 수 있습니다.");
        }

        foreach (var binding in slotBindings)
        {
            if (binding == null) continue;
            var key = Normalize(binding.Category);
            if (string.IsNullOrEmpty(key)) continue;

            if (_bindingLookup.ContainsKey(key))
            {
                Debug.LogWarning($"CustomizeManager: '{binding.Category}' 카테고리가 중복 등록되었습니다. 첫 번째 항목만 사용합니다.");
                continue;
            }

            if (binding.Renderer == null)
            {
                binding.Renderer = FindRendererByPrefix(binding.Category);
                if (binding.Renderer == null && !binding.UsesFacePicker)
                {
                    Debug.LogWarning($"CustomizeManager: '{binding.Category}'에 대응하는 SkinnedMeshRenderer를 찾지 못했습니다.");
                    continue;
                }
            }

            _bindingLookup.Add(key, binding);

            if (binding.Renderer != null)
            {
                _defaults[key] = new RendererSnapshot(binding.Renderer);
            }
        }

        if (facePicker == null && characterRoot != null)
        {
            facePicker = characterRoot.GetComponentInChildren<FacePicker>();
        }
    }

    private SkinnedMeshRenderer FindRendererByPrefix(string prefix)
    {
        if (characterRoot == null) return null;

        var lowered = prefix.ToLowerInvariant();
        return characterRoot
            .GetComponentsInChildren<SkinnedMeshRenderer>(true)
            .FirstOrDefault(r => r.transform.name.ToLowerInvariant().StartsWith(lowered));
    }

    private async Task EquipItemAsync(string categoryKey, string addressKey, SlotBinding binding)
    {
        ReleaseHandle(categoryKey);

        var handle = Addressables.LoadAssetAsync<GameObject>(addressKey);
        await handle.Task;

        if (handle.Status != AsyncOperationStatus.Succeeded)
        {
            Debug.LogError($"CustomizeManager: '{addressKey}' 어드레서블 로드 실패 (카테고리: {categoryKey}).");
            Addressables.Release(handle);
            ResetSlot(categoryKey);
            return;
        }

        _loadedHandles[categoryKey] = handle;

        if (binding.UsesFacePicker && facePicker != null)
        {
            if (!ApplyFace(handle.Result))
            {
                ResetSlot(categoryKey);
            }
            return;
        }

        if (binding.Renderer == null)
        {
            Debug.LogWarning($"CustomizeManager: '{categoryKey}'에 적용할 Renderer가 없습니다.");
            ResetSlot(categoryKey);
            return;
        }

        var sourceRenderer = handle.Result.GetComponentInChildren<SkinnedMeshRenderer>();
        if (sourceRenderer == null)
        {
            Debug.LogWarning($"CustomizeManager: '{addressKey}' 프리팹에서 SkinnedMeshRenderer를 찾지 못했습니다.");
            ResetSlot(categoryKey);
            return;
        }

        if (!_defaults.ContainsKey(categoryKey))
        {
            _defaults[categoryKey] = new RendererSnapshot(binding.Renderer);
        }

        binding.Renderer.sharedMesh = sourceRenderer.sharedMesh;
        binding.Renderer.sharedMaterials = sourceRenderer.sharedMaterials;
        if (sourceRenderer.sharedMesh != null)
        {
            binding.Renderer.localBounds = sourceRenderer.sharedMesh.bounds;
        }
        binding.Renderer.enabled = true;
    }

    // [--- 🛠️ 수정된 함수 🛠️ ---]
    private bool ApplyFace(GameObject sourcePrefab)
    {
        if (facePicker == null) return false;

        var meshRenderer = sourcePrefab.GetComponentInChildren<SkinnedMeshRenderer>();
        if (meshRenderer?.sharedMesh == null)
        {
            Debug.LogWarning("CustomizeManager: 얼굴 프리팹에서 Mesh를 찾지 못했습니다.");
            return false;
        }
        
        // 스크립트의 기존 이름 파싱 로직을 그대로 사용합니다.
        // (e.g., Prefab 안의 Mesh 이름이 'Face_Male_Default' 형식이어야 함)
        var nameSections = meshRenderer.sharedMesh.name.Split('_');
        if (nameSections.Length >= 3)
        {
            // 'CharacterCustomizationTool.Extensions'가 없으면 .ToCapital()이 작동하지 않을 수 있습니다.
            // .ToCapital()이 없다면, Enum.TryParse의 세 번째 인자인 'true' (ignoreCase)로 대체합니다.
            var faceToken = nameSections[2]; // .ToCapital() 제거

            // 'FaceType' Enum에서 일치하는 이름을 찾습니다 (대소문자 무시)
            if (Enum.TryParse(faceToken, true, out FaceType faceType))
            {
                // FacePicker가 이 얼굴을 알고 있는지 확인
                if (facePicker.HasFace(faceType))
                {
                    // 얼굴 선택!
                    facePicker.PickFace(faceType);
                    return true;
                }
                else
                {
                    Debug.LogWarning($"CustomizeManager: FacePicker에 {faceType}이(가) 없습니다. (메쉬 이름: {meshRenderer.sharedMesh.name})");
                    return false;
                }
            }
        }

        Debug.LogWarning($"CustomizeManager: FaceType 파싱 실패 ({meshRenderer.sharedMesh.name}). 이름 형식을 (Face_Gender_FaceType) 확인하세요.");
        return false;
    }

    private void ResetSlot(string categoryKey)
    {
        ReleaseHandle(categoryKey);

        if (!_bindingLookup.TryGetValue(categoryKey, out var binding))
        {
            return;
        }

        if (binding.ClearWhenMissing)
        {
            if (binding.Renderer != null)
            {
                binding.Renderer.sharedMesh = null;
                binding.Renderer.sharedMaterials = Array.Empty<Material>();
                binding.Renderer.enabled = false;
            }
            return;
        }

        if (_defaults.TryGetValue(categoryKey, out var snapshot) && binding.Renderer != null)
        {
            snapshot.Restore(binding.Renderer);
            binding.Renderer.enabled = true;
        }
    }

    private void ReleaseHandle(string categoryKey)
    {
        if (_loadedHandles.TryGetValue(categoryKey, out var handle))
        {
            if (handle.IsValid())
            {
                Addressables.Release(handle);
            }
            _loadedHandles.Remove(categoryKey);
        }
    }

    private void ReleaseAllHandles()
    {
        foreach (var handle in _loadedHandles.Values)
        {
            if (handle.IsValid())
            {
                Addressables.Release(handle);
            }
        }
        _loadedHandles.Clear();
    }

    private static string Normalize(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? string.Empty : value.Trim().ToLowerInvariant();
    }

    [Serializable]
    private class RendererSnapshot
    {
        private readonly Mesh _mesh;
        private readonly Material[] _materials;
        private readonly Bounds _bounds;

        public RendererSnapshot(SkinnedMeshRenderer renderer)
        {
            _mesh = renderer.sharedMesh;
            _materials = renderer.sharedMaterials.ToArray();
            _bounds = renderer.localBounds;
        }

        public void Restore(SkinnedMeshRenderer renderer)
        {
            renderer.sharedMesh = _mesh;
            renderer.sharedMaterials = _materials;
            renderer.localBounds = _bounds;
        }
    }

    [Serializable]
    private class SlotBinding
    {
        [SerializeField] private string category;
        [SerializeField] private SkinnedMeshRenderer renderer;
        [SerializeField] private bool usesFacePicker;

        public string Category => category;
        public SkinnedMeshRenderer Renderer { get => renderer; set => renderer = value; }
        [SerializeField] private bool clearWhenMissing = true;

        public bool UsesFacePicker => usesFacePicker;
        public bool ClearWhenMissing => clearWhenMissing;
    }
}