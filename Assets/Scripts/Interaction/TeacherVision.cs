using UnityEngine;
using Photon.Pun;

public class TeacherVision : MonoBehaviourPunCallbacks
{
    [Header("Settings")]
    public Color highlightColor = Color.green;
    public Color outlineColor = Color.black;
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

    // ReSharper disable Unity.PerformanceAnalysis
    private System.Collections.IEnumerator ScanRoutine()
    {
        WaitForSeconds wait = new WaitForSeconds(scanInterval);
        while (true)
        {
            ApplyHighlights();
            yield return wait;
        }
        // ReSharper disable once IteratorNeverReturns
    }

    public void ApplyHighlights()
    {
        // Find all objects in Layer 6 (Interactable)
        int layer = LayerMask.NameToLayer("Interactable");
        if (layer == -1) layer = 6;

        // Use FindObjectsByType for broader compatibility and safety
        GameObject[] allObjects = FindObjectsByType<GameObject>(FindObjectsSortMode.None);
        foreach (GameObject obj in allObjects)
        {
            if (obj.layer == layer)
            {
                if (obj.GetComponent<InteractableHighlighter>() == null)
                {
                    var highlighter = obj.AddComponent<InteractableHighlighter>();
                    highlighter.highlightColor = highlightColor;
                    highlighter.outlineColor = outlineColor;
                }
            }
        }
    }
}

public class InteractableHighlighter : MonoBehaviour
{
    private static readonly int ZWrite = Shader.PropertyToID("_ZWrite");
    private static readonly int ZTest = Shader.PropertyToID("_ZTest");
    public Color highlightColor = Color.green;
    public Color outlineColor = Color.black;
    
    private GameObject _mainCube;
    private GameObject _outlineCube;
    
    private Material _highlightMaterial;
    private Material _outlineMaterial;

    private void OnEnable()
    {
        CreateIndicator();
    }

    private void OnDisable()
    {
        Cleanup();
    }

    private void OnDestroy()
    {
        Cleanup();
    }

    private void Cleanup()
    {
        if (_mainCube != null) Destroy(_mainCube);
        if (_outlineCube != null) Destroy(_outlineCube);
        if (_highlightMaterial != null) Destroy(_highlightMaterial);
        if (_outlineMaterial != null) Destroy(_outlineMaterial);
    }

    private void CreateIndicator()
    {
        Cleanup(); // Ensure clean state

        // 1. Create Main Cube (Green)
        _mainCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        SetupCube(_mainCube, highlightColor, 0.1f, 0);

        // 2. Create Outline Cube (Black, slightly larger)
        _outlineCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        SetupCube(_outlineCube, outlineColor, 0.13f, 1);
    }

    private void SetupCube(GameObject cube, Color color, float scale, int renderQueueOffset)
    {
        Collider col = cube.GetComponent<Collider>();
        if (col != null) Destroy(col);

        cube.transform.localScale = new Vector3(scale, scale, scale);
        // Set layer to Default or Ignore Raycast to prevent self-detection if TeacherVision scans everything
        cube.layer = 0; 

        Material mat = CreateZTestAlwaysMaterial(color);
        mat.renderQueue = 3000 + renderQueueOffset;

        Renderer rend = cube.GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material = mat;
            rend.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            rend.receiveShadows = false;
        }
    }

    private Material CreateZTestAlwaysMaterial(Color color)
    {
        Shader shader = Shader.Find("Hidden/Internal-Colored");
        if (shader == null) shader = Shader.Find("GUI/Text Shader");
        if (shader == null) shader = Shader.Find("Unlit/Color");

        Material mat = new Material(shader);
        mat.color = color;
        mat.SetInt(ZTest, (int)UnityEngine.Rendering.CompareFunction.Always);
        mat.SetInt(ZWrite, 0);
        mat.EnableKeyword("_ALPHABLEND_ON");
        
        return mat;
    }

    private void Update()
    {
        if (!_mainCube || !_outlineCube) return;

        Vector3 targetPos = transform.position + Vector3.up * 2.0f;

        _mainCube.transform.position = targetPos;
        _outlineCube.transform.position = targetPos;

        _mainCube.transform.rotation = Quaternion.identity;
        _outlineCube.transform.rotation = Quaternion.identity;
    }
}
