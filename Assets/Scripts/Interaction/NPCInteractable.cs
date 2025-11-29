using UnityEngine;
using Interaction;
using Controller;

// Simple NPC interactable. Requires an InteractUI child object (like box/mission) and calls dialogue system on interact.
public class NPCInteractable : Interactable
{
    public string npcName; // optional override; if empty, use gameObject.name
    public string npcShownName; // optional display override for dialogue UI

    private GameObject interactUI;

    private void Start()
    {
        // If an interactUIPrefab (inherited from Interactable) is assigned, prefer prefab-instantiation.
        // Otherwise try find a child named InteractUI and disable it initially.
        if (interactUIPrefab == null)
        {
            var t = transform.Find("InteractUI");
            if (t != null) interactUI = t.gameObject;
            if (interactUI != null) interactUI.SetActive(false);
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            if (interactUI != null)
                interactUI.SetActive(true);
            else
                InteractUIShow(gameObject.name, transform);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            if (interactUI != null)
                interactUI.SetActive(false);
            else
                InteractUIHide();
        }
    }

    public override void Interact(PlayerInteraction player)
    {
        // hide the interact UI (support both child UI and prefab UI)
        if (interactUI != null)
            interactUI.SetActive(false);
        else
            InteractUIHide();

        string nameToUse = string.IsNullOrEmpty(npcName) ? gameObject.name : npcName;
        
        if (NPCDialogueSys.Instance != null)
        {
            // [수정] player.gameObject를 넘겨주어 대화 시스템이 플레이어의 컴포넌트(PlayerInteraction)를 찾을 수 있게 함
            NPCDialogueSys.Instance.ShowDialogue(nameToUse, player.gameObject, npcShownName);
        }
        else
        {
            Debug.LogWarning("[NPCInteractable] NPCDialogueSys.Instance is null. Make sure an NPCDialogueSys exists in the scene.");
        }
    }
}