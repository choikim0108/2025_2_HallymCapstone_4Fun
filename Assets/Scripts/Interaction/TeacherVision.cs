using UnityEngine;
using Photon.Pun;

public class TeacherVision : MonoBehaviourPunCallbacks
{
    [Header("Settings")]
    public Color highlightColor = new Color(0, 1, 0, 0.5f); // 50% 투명도의 녹색
    public float scanInterval = 2.0f;

    private Coroutine _scanCoroutine;

    public override void OnEnable()
    {
        base.OnEnable();
        // Check Role when enabled
        if (PhotonNetwork.IsConnected && PhotonNetwork.LocalPlayer != null)
        {
            if (PhotonNetwork.LocalPlayer.CustomProperties.TryGetValue("Role", out var role))
            {
                if (role != null && role.ToString() == "Teacher")
                {
                    if (_scanCoroutine != null) StopCoroutine(_scanCoroutine);
                    _scanCoroutine = StartCoroutine(ScanRoutine());
                }
            }
        }
    }

    public override void OnDisable()
    {
        base.OnDisable();
        if (_scanCoroutine == null) return;
        StopCoroutine(_scanCoroutine);
        _scanCoroutine = null;
    }

    private System.Collections.IEnumerator ScanRoutine()
    {
        WaitForSeconds wait = new WaitForSeconds(scanInterval);
        while (true)
        {
            ApplyHighlights();
            yield return wait;
        }
    }

    public void ApplyHighlights()
    {
        // Find all objects in Layer 6 (Interactable)
        int layer = LayerMask.NameToLayer("Interactable");
        if (layer == -1) layer = 6;

        GameObject[] allObjects = FindObjectsByType<GameObject>(FindObjectsSortMode.None);
        foreach (GameObject obj in allObjects)
        {
            if (obj.layer == layer)
            {
                if (obj.GetComponent<InteractableHighlighter>() == null)
                {
                    var highlighter = obj.AddComponent<InteractableHighlighter>();
                    highlighter.highlightColor = highlightColor;
                }
            }
        }
    }
}

public class InteractableHighlighter : MonoBehaviour
{
    private static readonly int ZWrite = Shader.PropertyToID("_ZWrite");
    private static readonly int ZTest = Shader.PropertyToID("_ZTest");
    private static readonly int SrcBlend = Shader.PropertyToID("_SrcBlend");
    private static readonly int DstBlend = Shader.PropertyToID("_DstBlend");
    
    public Color highlightColor = new Color(0, 1, 0, 0.5f);

    private Material[] _originalMaterials;
    private Material[] _overlayMaterials;
    private Renderer _renderer;

    private void OnEnable()
    {
        ApplyOverlay();
    }

    private void OnDisable()
    {
        RemoveOverlay();
    }

    private void OnDestroy()
    {
        RemoveOverlay();
    }

    private void ApplyOverlay()
    {
        _renderer = GetComponent<Renderer>();
        if (_renderer == null) return;

        _originalMaterials = _renderer.materials;
        _overlayMaterials = new Material[_originalMaterials.Length + 1];

        // 기존 머티리얼 복사
        for (int i = 0; i < _originalMaterials.Length; i++)
        {
            _overlayMaterials[i] = _originalMaterials[i];
        }

        // 오버레이 머티리얼 추가
        _overlayMaterials[_originalMaterials.Length] = CreateOverlayMaterial();
        _renderer.materials = _overlayMaterials;
    }

    private Material CreateOverlayMaterial()
    {
        Shader shader = Shader.Find("Hidden/Internal-Colored");
        if (shader == null) shader = Shader.Find("Standard");

        Material mat = new Material(shader);
        mat.color = highlightColor;
        
        // ZTest Always로 설정하여 항상 보이게 함
        mat.SetInt(ZTest, (int)UnityEngine.Rendering.CompareFunction.Always);
        mat.SetInt(ZWrite, 0);
        
        // 투명도 처리
        mat.SetInt(SrcBlend, (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt(DstBlend, (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.renderQueue = 3000;

        return mat;
    }

    private void RemoveOverlay()
    {
        if (_renderer != null && _originalMaterials != null)
        {
            _renderer.materials = _originalMaterials;
        }

        if (_overlayMaterials != null)
        {
            for (int i = _originalMaterials.Length; i < _overlayMaterials.Length; i++)
            {
                if (_overlayMaterials[i] != null)
                {
                    Destroy(_overlayMaterials[i]);
                }
            }
        }
    }
}