using UnityEngine;

namespace Interaction
{

    public abstract class Interactable : MonoBehaviour
    {
        [Header("Interactable Distance")] public float interactDistance = 2f;
        [Header("Interactable UI Prefab")] public GameObject interactUIPrefab;
        protected GameObject interactUIInstance;

        // 상호작용 함수 (상속받아 구현)
        public abstract void Interact();

        // UI 표시
        public virtual void ShowUI()
        {
            if (interactUIPrefab != null && interactUIInstance == null)
            {
                interactUIInstance = Instantiate(interactUIPrefab, transform.position + Vector3.up * 2f, Quaternion.identity);
            }
        }

        // UI 숨김
        public virtual void HideUI()
        {
            if (interactUIInstance != null)
            {
                Destroy(interactUIInstance);
                interactUIInstance = null;
            }
        }
    }
}
