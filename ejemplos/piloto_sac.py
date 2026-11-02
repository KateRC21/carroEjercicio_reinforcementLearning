"""Politica SAC ya entrenada. No aprende: solo ejecuta la decision guardada.

    python -m ejemplos.piloto_sac [pista] [--render]

El modelo de Algoritmo_solucion/sac_carro_v4.zip se entrena en el ovalo.
"""


"""
Diagnóstico de la política SAC entrenada.

Uso:
    python -m ejemplos.piloto_sac ovalo --render
"""

import sys
from pathlib import Path

import numpy as np
import carro_rl  # noqa: F401
import gymnasium as gym
from stable_baselines3 import SAC

RUTA_MODELO = (
    Path(__file__).resolve().parents[1]
    / "Algoritmo_solucion"
    / "sac_carro_v4"
)


if __name__ == "__main__":
    print("=== EJECUTANDO DIAGNÓSTICO SAC ===")
    print("Archivo:", Path(__file__).resolve())
    pista = next(
        (a for a in sys.argv[1:] if not a.startswith("--")),
        "ovalo",
    )
    modo = "human" if "--render" in sys.argv else None

    modelo = SAC.load(RUTA_MODELO, device="cpu")

    env = gym.make(
        "CarroPista-v0",
        pista=pista,
        render_mode=modo,
    )

    try:
        obs, info = env.reset(seed=0)

        print("Espacio de acciones:", env.action_space)
        print("Espacio de observaciones:", env.observation_space)
        print("Observación inicial:", np.round(obs, 4))
        print("\nIniciando diagnóstico...\n")

        total = 0.0
        paso = 0
        fin = False

        while not fin:
            accion, _ = modelo.predict(
                obs,
                deterministic=True,
            )

            obs, recompensa, term, trunc, info = env.step(accion)
            total += recompensa

            if paso < 20 or paso % 100 == 0:
                print(
                    f"Paso={paso:4d} | "
                    f"Acción={np.round(accion, 4)} | "
                    f"Obs={np.round(obs, 4)} | "
                    f"Recompensa={recompensa:.4f} | "
                    f"Progreso={info.get('progreso', 'N/D')} | "
                    f"Choque={info.get('choque', 'N/D')}"
                )

            fin = term or trunc
            paso += 1

        print("\n--- RESULTADO FINAL ---")
        print(f"Pista: {info.get('pista')}")
        print(f"Vuelta completada: {info.get('vuelta_completada')}")
        print(f"Choque: {info.get('choque')}")
        print(f"Progreso: {info.get('progreso')}")
        print(f"Tiempo: {info.get('tiempo')}")
        print(f"Retorno: {total:.2f}")
        print(f"Truncado: {trunc}")
        print(f"Pasos ejecutados: {paso}")

    finally:
        env.close()

