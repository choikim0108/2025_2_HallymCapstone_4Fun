using UnityEngine;
using Interaction;

public class PlayerInteraction : MonoBehaviour
{
    public float checkRadius = 2f; // 상호작용 감지 반경
    public LayerMask interactableLayer; // Interactable 오브젝트만 감지
    private Interactable currentTarget;

    void Update()
    {
        CheckInteractable();
        HandleInteractionInput();
    }

    void CheckInteractable()
    {
        Collider[] hits = Physics.OverlapSphere(transform.position, checkRadius, interactableLayer);
        Interactable nearest = null;
        float minDist = float.MaxValue;
        foreach (var hit in hits)
        {
            var interactable = hit.GetComponent<Interactable>();
            if (interactable != null)
            {
                float dist = Vector3.Distance(transform.position, interactable.transform.position);
                if (dist < interactable.interactDistance && dist < minDist)
                {
                    nearest = interactable;
                    minDist = dist;
                }
            }
        }

        if (nearest != null)
        {
            if (currentTarget != nearest)
            {
                if (currentTarget != null) currentTarget.HideUI();
                currentTarget = nearest;
                currentTarget.ShowUI(currentTarget.gameObject.name, currentTarget.transform);
            }
        }
        else
        {
            if (currentTarget != null)
            {
                currentTarget.HideUI();
                currentTarget = null;
            }
        }
    }

    void HandleInteractionInput()
    {
        if (currentTarget != null && Input.GetKeyDown(KeyCode.F))
        {
            currentTarget.Interact();
        }
    }
}
