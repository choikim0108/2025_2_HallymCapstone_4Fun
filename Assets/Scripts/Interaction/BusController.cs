using UnityEngine;
using Photon.Pun;
using Photon.Realtime; // Player 타입을 쓰기 위해 필요
using System.Collections.Generic;

public class BusController : MonoBehaviourPun
{
    public float moveSpeed = 10f;
    public float turnSpeed = 50f;
    public Transform exitPoint;

    private int driverViewID = -1;
    private List<int> passengerViewIDs = new List<int>();

    void Awake()
    {
        if (exitPoint == null)
        {
            exitPoint = new GameObject("ExitPoint").transform;
            exitPoint.SetParent(this.transform);
            exitPoint.localPosition = new Vector3(-2.5f, 0, 0);
        }
    }

    void Update()
    {
        // [중요] 소유권을 가진 사람(운전자)만 이동 로직을 실행하고, 
        // 그 결과는 PhotonTransformView를 통해 자동으로 동기화됩니다.
        if (photonView.IsMine && driverViewID != -1)
        {
            // 현재 운전자가 "나"인지 한 번 더 확인 (이중 체크)
            var driverPhotonView = PhotonView.Find(driverViewID);
            if (driverPhotonView != null && driverPhotonView.IsMine)
            {
                HandleMovement();
            }
        }
    }

    private void HandleMovement()
    {
        float moveInput = Input.GetAxis("Vertical");
        float turnInput = Input.GetAxis("Horizontal");

        if (Mathf.Abs(moveInput) < 0.01f)
        {
            turnInput = 0f;
        }
        else
        {
            if (moveInput < 0)
            {
                turnInput = -turnInput;
            }
        }

        transform.Translate(Vector3.forward * moveInput * moveSpeed * Time.deltaTime);
        transform.Rotate(Vector3.up * turnInput * turnSpeed * Time.deltaTime);
    }

    // [RPC] 플레이어의 탑승 요청 처리
    [PunRPC]
    public void BoardRequestRPC(int playerViewID)
    {
        if (!PhotonNetwork.IsMasterClient) return;

        if (driverViewID == playerViewID || passengerViewIDs.Contains(playerViewID)) return;

        if (driverViewID == -1) // 운전석이 비어있으면
        {
            // [핵심] 운전자에게 버스의 소유권을 넘깁니다.
            PhotonView playerPv = PhotonView.Find(playerViewID);
            if (playerPv != null)
            {
                // 소유권 이전 (이 코드는 MasterClient에서 실행되므로 권한이 있습니다)
                photonView.TransferOwnership(playerPv.Owner);
            }

            photonView.RPC("SetDriverRPC", RpcTarget.All, playerViewID);
        }
        else // 운전석이 차있으면
        {
            photonView.RPC("AddPassengerRPC", RpcTarget.All, playerViewID);
        }
    }

    // [RPC] 플레이어의 하차 요청 처리
    [PunRPC]
    public void GetOffRequestRPC(int playerViewID)
    {
        if (!PhotonNetwork.IsMasterClient) return;

        if (driverViewID == playerViewID) // 운전자가 내리는 경우
        {
            // 모든 탑승자 하차
            List<int> allRiders = new List<int>(passengerViewIDs);
            allRiders.Add(driverViewID);

            foreach (var riderID in allRiders)
            {
                photonView.RPC("EjectPlayerRPC", RpcTarget.All, riderID);
            }

            // [핵심] 운전자가 내리면 소유권을 다시 가져옵니다 (MasterClient나 Scene으로)
            // RequestOwnership()은 소유권을 뺏어오는 것이고, TransferOwnership은 넘겨주는 것입니다.
            // 여기서는 MasterClient가 다시 소유권을 가져갑니다.
            if (photonView.Owner != PhotonNetwork.MasterClient)
            {
                photonView.TransferOwnership(PhotonNetwork.MasterClient);
            }

            photonView.RPC("ResetBusRPC", RpcTarget.All);
        }
        else if (passengerViewIDs.Contains(playerViewID))
        {
            photonView.RPC("EjectPlayerRPC", RpcTarget.All, playerViewID);
        }
    }

    [PunRPC]
    private void SetDriverRPC(int playerViewID)
    {
        driverViewID = playerViewID;
        var playerView = PhotonView.Find(playerViewID);
        
        // IsMine 체크: 이 코드를 실행하는 클라이언트가 "운전자 본인"일 때만 탑승 처리
        if (playerView != null && playerView.IsMine)
        {
            playerView.GetComponent<PlayerInteraction>().OnBoardedAsDriver(this.gameObject);
        }
    }

    [PunRPC]
    private void AddPassengerRPC(int playerViewID)
    {
        if (!passengerViewIDs.Contains(playerViewID))
        {
            passengerViewIDs.Add(playerViewID);
            var playerView = PhotonView.Find(playerViewID);
            if (playerView != null && playerView.IsMine)
            {
                playerView.GetComponent<PlayerInteraction>().OnBoardedAsPassenger(this.gameObject);
            }
        }
    }

    [PunRPC]
    private void EjectPlayerRPC(int playerViewID)
    {
        if (driverViewID == playerViewID) driverViewID = -1;
        if (passengerViewIDs.Contains(playerViewID)) passengerViewIDs.Remove(playerViewID);

        var playerView = PhotonView.Find(playerViewID);
        if (playerView != null && playerView.IsMine)
        {
            playerView.GetComponent<PlayerInteraction>().OnGetOffBus(exitPoint.position);
        }
    }

    [PunRPC]
    private void ResetBusRPC()
    {
        driverViewID = -1;
        passengerViewIDs.Clear();
    }
}