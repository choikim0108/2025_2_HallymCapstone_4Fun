using UnityEngine;

namespace Interaction
{

    public abstract class Interactable : MonoBehaviour
    {
        [Header("Interactable Distance")] public float interactDistance = 2f;
        [Header("Interactable UI Prefab")] public GameObject interactUIPrefab;
    protected GameObject interactUIInstance;
    private float uiShowTimestamp = -1f;
    private float minShowTime = 0.2f; // 최소 표시 시간(초)

        // 상호작용 함수 (상속받아 구현)
        public abstract void Interact();

        // UI 표시 (오브젝트 이름 포함)
        public virtual void ShowUI(string objectName = "", Transform targetTransform = null)
        {
            Debug.Log($"[Interactable] ShowUI called for {gameObject.name}");
            if (interactUIPrefab != null && interactUIInstance == null)
            {
                Canvas canvas = GameObject.FindFirstObjectByType<Canvas>();
                Transform parent = null;
                if (canvas != null)
                    parent = canvas.transform;

                interactUIInstance = Instantiate(interactUIPrefab, parent);
                Debug.Log($"[Interactable] InteractUI instantiated for {gameObject.name}");
                var ui = interactUIInstance.GetComponent<InteractUI>();
                if (ui != null)
                {
                    ui.SetMessage(objectName);
                    if (targetTransform != null)
                        ui.SetTarget(targetTransform, canvas);
                }
                uiShowTimestamp = Time.time;
            }
            else if (interactUIInstance != null)
            {
                Debug.Log($"[Interactable] ShowUI called but UI already exists for {gameObject.name}");
            }
        }

        // UI 숨김
        public virtual void HideUI()
        {
            Debug.Log($"[Interactable] HideUI called for {gameObject.name}");
            // 최소 표시 시간 보장
            if (interactUIInstance != null)
            {
                if (uiShowTimestamp > 0 && (Time.time - uiShowTimestamp) < minShowTime)
                {
                    float delay = minShowTime - (Time.time - uiShowTimestamp);
                    if (delay > 0)
                    {
                        Debug.Log($"[Interactable] Destroy scheduled for {gameObject.name} after {delay} seconds");
                        Destroy(interactUIInstance, delay);
                    }
                    else
                    {
                        Debug.Log($"[Interactable] Destroy immediately for {gameObject.name}");
                        Destroy(interactUIInstance);
                    }
                }
                else
                {
                    Debug.Log($"[Interactable] Destroy immediately for {gameObject.name}");
                    Destroy(interactUIInstance);
                }
                interactUIInstance = null;
                uiShowTimestamp = -1f;
            }
            else
            {
                Debug.Log($"[Interactable] HideUI called but no UI instance for {gameObject.name}");
            }
        }
    }
}
