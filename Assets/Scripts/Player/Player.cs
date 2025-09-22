using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Player : MonoBehaviour
{
    
    public float MoveSpeed { get { return moveSpeed; } }

    [SerializeField] protected float moveSpeed;

    public void OnUpdateStat(float moveSpeed)
    {
        this.moveSpeed = moveSpeed;
    }
}