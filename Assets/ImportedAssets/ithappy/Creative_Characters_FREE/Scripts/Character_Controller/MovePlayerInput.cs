using Photon.Pun;
using UnityEngine;

namespace Controller
{
    [RequireComponent(typeof(CharacterMover))]
    public class MovePlayerInput : MonoBehaviourPun
    {
        [Header("Character")]
        [SerializeField]
        private string m_HorizontalAxis = "Horizontal";
        [SerializeField]
        private string m_VerticalAxis = "Vertical";
        [SerializeField]
        private string m_JumpButton = "Jump";
        [SerializeField]
        private KeyCode m_RunKey = KeyCode.LeftShift;

        [Header("Camera")]
        [SerializeField]
        private PlayerCamera m_Camera;
        [SerializeField]
        private string m_MouseX = "Mouse X";
        [SerializeField]
        private string m_MouseY = "Mouse Y";
        [SerializeField]
        private string m_MouseScroll = "Mouse ScrollWheel";

        private CharacterMover m_Mover;

        private Vector2 m_Axis;
        private bool m_IsRun;
        private bool m_IsJump;

        private Vector3 m_Target;
        private Vector2 m_MouseDelta;
        private float m_Scroll;

        private bool inputEnabled = true;
        
        private void Awake()
        {
            m_Mover = GetComponent<CharacterMover>();

            if(m_Camera == null ) 
            {
                m_Camera = Camera.main == null ? null : Camera.main.GetComponent<PlayerCamera>();
            }
            if(m_Camera != null) {
                m_Camera.SetPlayer(transform);
                Debug.Log($"[MovePlayerInput] m_Camera.SetPlayer({transform.name}) 호출", m_Camera);
            } else {
                Debug.LogWarning("[MovePlayerInput] m_Camera가 할당되지 않음", this);
            }
        }


        private void Update()
        {
            if (!inputEnabled) return;
            
            GatherInput();
            SetInput();
            
        }

        public void SetInputEnabled(bool enabled)
        {
            inputEnabled = enabled;
        }

        public void GatherInput()
        {
            m_Axis = new Vector2(Input.GetAxis(m_HorizontalAxis), Input.GetAxis(m_VerticalAxis));
            m_IsRun = Input.GetKey(m_RunKey);
            m_IsJump = Input.GetButton(m_JumpButton);

            // 카메라의 forward(y축만) 벡터를 m_Target에 전달
            if (m_Camera != null)
            {
                Vector3 camForward = m_Camera.transform.forward;
                camForward.y = 0f;
                camForward.Normalize();
                m_Target = camForward;
            }
            else
            {
                m_Target = Vector3.forward;
            }
            m_MouseDelta = new Vector2(Input.GetAxis(m_MouseX), Input.GetAxis(m_MouseY));
            m_Scroll = Input.GetAxis(m_MouseScroll);
        }

        public void BindMover(CharacterMover mover)
        {
            m_Mover = mover;
        }

        public void SetInput()
        {
            if (m_Mover != null)
            {
                m_Mover.SetInput(in m_Axis, in m_Target, in m_IsRun, m_IsJump);
            }

            if (m_Camera != null)
            {
                m_Camera.SetInput(in m_MouseDelta, m_Scroll);
            }
        }

        public void ResetInputState()
        {
            m_Axis = Vector2.zero;
            m_IsRun = false;
            m_IsJump = false;
            m_MouseDelta = Vector2.zero;
            m_Scroll = 0f;

            // CharacterMover에도 초기화된 입력 전달
            if (m_Mover != null) m_Mover.SetInput(in m_Axis, in m_Target, in m_IsRun, m_IsJump);

            // 카메라에도 초기화된 입력 전달
            if (m_Camera != null) m_Camera.SetInput(in m_MouseDelta, m_Scroll);
        }
        
    }
}