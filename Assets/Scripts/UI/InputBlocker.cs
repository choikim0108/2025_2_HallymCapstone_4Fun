using System;
using UnityEngine;
using Controller;
using Photon.Pun;

namespace UI
{
    /// <summary>
    /// 특정 UI가 활성화되어 있을 때 플레이어의 입력을 차단합니다.
    /// </summary>
    public class InputBlocker : MonoBehaviour
    {
        [Header("Target UIs")]
        [SerializeField]
        [Tooltip("이 UI들 중 하나라도 활성화되어 있으면 플레이어 입력을 차단합니다")]
        private GameObject[] targetUIs;

        [Header("Player Components")]
        [SerializeField]
        [Tooltip("입력을 차단할 플레이어의 MovePlayerInput 컴포넌트")]
        private MovePlayerInput playerInput;

        [SerializeField]
        [Tooltip("카메라 조작을 차단할 PlayerCamera 컴포넌트")]
        private PlayerCamera playerCamera;

        private bool _wasUIActive;
        private bool _originalCursorState;

        [Obsolete("Obsolete")]
        private void Start()
        {
            // 플레이어 컴포넌트가 설정되지 않은 경우 로컬 플레이어만 자동으로 찾기
            if (playerInput == null)
            {
                playerInput = FindLocalPlayerInput();
                if (playerInput == null)
                {
                    Debug.LogWarning("[InputBlocker] MovePlayerInput을 찾을 수 없습니다.", this);
                }
            }

            if (playerCamera == null)
            {
                playerCamera = FindLocalPlayerCamera();
                if (playerCamera == null)
                {
                    Debug.LogWarning("[InputBlocker] PlayerCamera를 찾을 수 없습니다.", this);
                }
            }
        }

        [Obsolete("Obsolete")]
        private MovePlayerInput FindLocalPlayerInput()
        {
            var allInputs = FindObjectsOfType<MovePlayerInput>();
            foreach (var input in allInputs)
            {
                var photonView = input.GetComponent<PhotonView>();
                if (photonView != null && photonView.IsMine)
                {
                    return input;
                }
            }
            return null;
        }

        [Obsolete("Obsolete")]
        private PlayerCamera FindLocalPlayerCamera()
        {
            var allCameras = FindObjectsOfType<PlayerCamera>();
            foreach (var cam in allCameras)
            {
                var photonView = cam.GetComponent<PhotonView>();
                if (photonView != null && photonView.IsMine)
                {
                    return cam;
                }
            }
            return null;
        }

        private void Update()
        {
            if (targetUIs == null || targetUIs.Length == 0) return;

            bool isAnyUIActive = IsAnyUIActive();

            // UI 상태가 변경되었을 때만 처리
            if (isAnyUIActive != _wasUIActive)
            {
                if (isAnyUIActive)
                {
                    BlockInput();
                }
                else
                {
                    UnblockInput();
                }

                _wasUIActive = isAnyUIActive;
            }
        }

        private bool IsAnyUIActive()
        {
            foreach (var ui in targetUIs)
            {
                // activeInHierarchy는 오브젝트와 모든 부모가 활성화되어 있을 때만 true
                if (ui && ui.activeInHierarchy)
                {
                    return true;
                }
            }
            return false;
        }


        private void BlockInput()
        {
            // 플레이어 입력 상태 초기화 및 차단
            if (playerInput != null)
            {
                playerInput.ResetInputState(); // 먼저 입력 상태 초기화
                playerInput.SetInputEnabled(false);
            }

            // 카메라 입력 상태 초기화 및 차단
            if (playerCamera != null)
            {
                playerCamera.ResetInputState(); // 먼저 입력 상태 초기화
                playerCamera.SetInputEnabled(false);
            }

            // 커서 표시 및 잠금 해제
            _originalCursorState = Cursor.visible;
            Cursor.visible = true;
            Cursor.lockState = CursorLockMode.None;

            // 활성화된 UI 찾기
            string activeUIName = GetActiveUINames();
            Debug.Log($"[InputBlocker] 플레이어 입력 차단됨 - 활성화된 UI: {activeUIName}", this);
        }


        private void UnblockInput()
        {
            // 플레이어 입력 처리 활성화
            if (playerInput != null)
            {
                playerInput.SetInputEnabled(true);
            }

            // 카메라 입력 처리 활성화
            if (playerCamera != null)
            {
                playerCamera.SetInputEnabled(true);
            }

            // 커서 상태 복원
            Cursor.visible = _originalCursorState;
            Cursor.lockState = CursorLockMode.Locked;

            Debug.Log("[InputBlocker] 플레이어 입력 활성화됨 - 모든 UI 비활성화", this);
        }

        private string GetActiveUINames()
        {
            if (targetUIs == null || targetUIs.Length == 0)
                return "None";

            var activeNames = new System.Collections.Generic.List<string>();
            foreach (var ui in targetUIs)
            {
                if (ui != null && ui.activeSelf)
                {
                    activeNames.Add(ui.name);
                }
            }

            return activeNames.Count > 0 ? string.Join(", ", activeNames) : "None";
        }


        private void OnDisable()
        {
            // 스크립트가 비활성화될 때 입력 복원
            if (_wasUIActive)
            {
                UnblockInput();
            }
        }
    }
}
