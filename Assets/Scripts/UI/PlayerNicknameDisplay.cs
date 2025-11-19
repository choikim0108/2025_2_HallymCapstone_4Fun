using Photon.Pun;
using UnityEngine;
using TMPro; // TextMeshPro를 사용하려면 이 네임스페이스가 필요합니다.

public class PlayerNicknameDisplay : MonoBehaviour
{
    public TextMeshProUGUI nicknameText; // 닉네임을 표시할 TextMeshPro UI
    private PhotonView photonView;

    void Start()
    {
        photonView = GetComponent<PhotonView>();

        // 닉네임 텍스트가 할당되었는지 확인
        if (nicknameText == null)
        {
            Debug.LogError("Nickname Text(TextMeshProUGUI)가 할당되지 않았습니다.", this);
            return;
        }

        // PhotonView가 있고, 그 소유자(Owner)가 유효한지 확인
        if (photonView != null && photonView.Owner != null)
        {
            // 이 오브젝트의 소유자(로컬 플레이어든 리모트 플레이어든)의 닉네임을 가져와 설정
            nicknameText.text = photonView.Owner.NickName;
        }
        else
        {
            // PhotonView가 없거나 소유자가 없는 경우 (예: 씬에 미리 배치된 오브젝트)
            // 혹은 아직 소유자 정보가 동기화되기 전
            if (photonView == null)
            {
                Debug.LogError("PhotonView 컴포넌트가 없습니다.", this);
            }
            else
            {
                // IsMine은 로컬 플레이어 소유일 때만 true
                if (photonView.IsMine)
                {
                    // 로컬 플레이어의 경우 PhotonNetwork.NickName에서 직접 가져올 수 있습니다.
                    nicknameText.text = PhotonNetwork.NickName;
                }
                else
                {
                    // 아직 Owner 정보가 준비되지 않았을 수 있습니다.
                    // 이 경우 닉네임이 나중에 설정되거나, 
                    // OnPhotonPlayerPropertiesChanged 콜백 등을 이용해 갱신해야 할 수도 있습니다.
                    // 하지만 대부분의 경우 Start() 시점 이전에 Owner 정보가 설정됩니다.
                    Debug.LogWarning("아직 PhotonView.Owner 정보가 준비되지 않았습니다.", this);
                }
            }
        }
    }
}