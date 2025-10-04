using UnityEngine;
using Photon.Pun;

namespace Controller
{
    // 이 스크립트는 DefaultCharacter 프리팹 루트에 추가하세요.
    // 내 캐릭터가 아니면(타인 캐릭터면) 입력/카메라/오디오 등 모든 조작 관련 컴포넌트를 자동으로 비활성화합니다.
    public class PhotonLocalPlayerOnly : MonoBehaviourPun
    {
        void Awake()
        {
            if (photonView != null && !photonView.IsMine)
            {
                // 입력/조작 관련 컴포넌트 비활성화
                var moveInput = GetComponent<MovePlayerInput>();
                if (moveInput != null) moveInput.enabled = false;
                var mover = GetComponent<CharacterMover>();
                if (mover != null) mover.enabled = false;

                // 카메라 관련 컴포넌트 비활성화 (자식 MainCamera)
                var cameraObj = transform.Find("CameraContainer/MainCamera");
                if (cameraObj != null)
                {
                    cameraObj.gameObject.SetActive(false);
                }
            }
        }
    }
}
