using UnityEngine;
using Photon.Pun;
using Interaction;

public class MissionGiverInteractable : Interactable
{
    // 박스를 생성하지 않으므로 스폰 포인트 변수는 더 이상 필요하지 않아 제거하거나 주석 처리합니다.
    // public Transform boxSpawnPoint; 
    // public string boxesResourcePath = "Boxes"; 

    public override void Interact(PlayerInteraction player)
    {
        // 1) 랜덤 박스 이름 선택
        string[] boxNames = { "Box_Normal", "Box_Fragile", "Box_Heavy", "Box_Explosive", "Box_Frozen" };
        string selectedBoxName = boxNames[Random.Range(0, boxNames.Length)];

        /* [수정됨] 
           기존의 프리팹 로드 및 PhotonNetwork.Instantiate (박스 생성) 로직을 모두 삭제했습니다.
           이제 물리적인 박스는 생성되지 않고, 미션 텍스트만 갱신됩니다.
        */

        // 2) 플레이어의 UI에 미션 정보 표시 (로컬 클라이언트만)
        if (player.photonView.IsMine) 
        {
            if (MissionInfoUI.Instance != null)
            {
                // UI에 랜덤으로 선택된 박스 이름을 전달하여 미션 갱신
                MissionInfoUI.Instance.ShowMission(selectedBoxName);
                Debug.Log($"[MissionGiver] 새로운 미션 부여됨: {selectedBoxName}");
            }
            else
            {
                Debug.LogWarning("씬에 MissionInfoUI가 없습니다.");
            }
        }
    }
}