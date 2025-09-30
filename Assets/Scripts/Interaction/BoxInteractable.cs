using UnityEngine;
using Interaction;

public class BoxInteractable : Interactable
{
    public override void Interact()
    {
        // 여기에 Box_1 상호작용 동작을 구현하세요.
        Debug.Log($"{gameObject.name}과(와) 상호작용!");
    }
}
