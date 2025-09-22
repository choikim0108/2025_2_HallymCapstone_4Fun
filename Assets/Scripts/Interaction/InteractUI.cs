using UnityEngine;
using UnityEngine.UI;

namespace Interaction
{

    public class InteractUI : MonoBehaviour
    {
        [Header("UI text")] public Text messageText;

        public void SetMessage(string msg)
        {
            if (messageText != null)
                messageText.text = msg;
        }
    }
}
