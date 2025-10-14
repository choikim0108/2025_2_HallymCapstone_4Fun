using UnityEngine;
using Photon.Pun;
using Interaction;
using System;

public class PlayerInteraction : MonoBehaviourPun
{
    public float checkRadius = 2f; // 상호작용 감지 반경
    public LayerMask interactableLayer; // Interactable 오브젝트만 감지
    public Interactable currentTarget;

    public string equippedTarget;
    public GameObject equippedBoxUI; // BoxInteractable에서 직접 제어할 수 있도록 참조 추가

    void Awake()
    {

    }

    void Update()
    {
        if (!photonView.IsMine) return; // 내 플레이어만 처리

        CheckInteractable();
        HandleInteractionInput();
    }

    void CheckInteractable()
    {
        Collider[] hits = Physics.OverlapSphere(transform.position, checkRadius, interactableLayer);
        //Debug.Log($"[PlayerInteraction] 감지된 Collider 개수: {hits.Length}");
        Interactable nearest = null;
        float minDist = float.MaxValue;
        foreach (var hit in hits)
        {
            //Debug.Log($"[PlayerInteraction] 감지된 Collider: {hit.gameObject.name}, Layer: {LayerMask.LayerToName(hit.gameObject.layer)}");
            var interactable = hit.GetComponent<Interactable>();
            if (interactable != null)
            {
                //Debug.Log($"[PlayerInteraction] Interactable 컴포넌트 발견: {interactable.gameObject.name}, interactDistance: {interactable.interactDistance}");
                float dist = Vector3.Distance(transform.position, interactable.transform.position);
                //Debug.Log($"[PlayerInteraction] Player와 거리: {dist}");
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
                if (currentTarget != null) currentTarget.InteractUIHide();
                currentTarget = nearest;
                currentTarget.InteractUIShow(currentTarget.gameObject.name, currentTarget.transform);

                // 서버와 동기화 (PhotonView null 체크)
                var nearestPhotonView = nearest.GetComponent<PhotonView>();
                if (nearestPhotonView != null)
                {
                    photonView.RPC("SetCurrentInteractableRPC", RpcTarget.All, nearestPhotonView.ViewID);
                }
                else
                {
                    photonView.RPC("SetCurrentInteractableRPC", RpcTarget.All, -1);
                }
            }
        }
        else
        {
            if (currentTarget != null)
            {
                currentTarget.InteractUIHide();
                currentTarget = null;
                photonView.RPC("SetCurrentInteractableRPC", RpcTarget.All, -1);
            }
        }
    }

    void HandleInteractionInput()
    {
        if (currentTarget != null && Input.GetKeyDown(KeyCode.F))
        {
            currentTarget.Interact(this);
            Debug.Log($"[PlayerInteraction] Interacted with {currentTarget.gameObject.name}");
        }
    }

    [PunRPC]
    public void SetCurrentInteractableRPC(int viewID)
    {
        if (viewID == -1)
        {
            currentTarget = null;
        }
        else
        {
            var obj = PhotonView.Find(viewID);
            if (obj != null)
            {
                currentTarget = obj.GetComponent<Interactable>();
            }
        }
    }
}
