# SPH Fluid Simulation — GPU (Metal)

Simulation de fluide par **Smoothed Particle Hydrodynamics (SPH)** entièrement exécutée sur GPU via **Metal Compute Shaders** (macOS).
~6 000 particules simulées en temps réel, rendu par sphere impostors instanciés.

---

## Aperçu

| Composant | Détail |
|-----------|--------|
| Méthode numérique | SPH (Smoothed Particle Hydrodynamics) |
| Intégration | Euler symplectique (vitesse → position) |
| Recherche de voisins | Grille uniforme avec hachage spatial |
| Rendu | Sphere impostors billboardés (1 draw call instancié) |
| GPU API | Metal 3 (macOS 13+) |
| Langage | Swift 5.9 + MSL (Metal Shading Language) |

---

## Physique — comment ça marche

### SPH en bref

SPH discrétise le fluide en **particules porteuses de masse**.
Chaque grandeur physique est estimée en convoluant les valeurs voisines avec une **fonction noyau** W(r, h) où h est le rayon de lissage.

```
A(x) = Σⱼ mⱼ · Aⱼ/ρⱼ · W(|x – xⱼ|, h)
```

### Noyaux utilisés

| Noyau | Usage | Formule |
|-------|-------|---------|
| **Poly6** | Densité | `315/(64π h⁹) · (h²–r²)³` |
| **Spiky** ∇ | Force de pression | `–45/(π h⁶) · (h–r)² · r̂` |
| **Viscosité** ∇² | Force de viscosité | `45/(π h⁶) · (h–r)` |

### Équation d'état

La pression est calculée via l'équation de Tait (fluide faiblement compressible) :

```
P = k · (ρ – ρ₀)
```

- `k` : constante de rigidité (raideur de pression)
- `ρ₀` : densité de repos (1000 kg/m³ pour l'eau)

### Forces

```
F_total = F_pression + F_viscosité + F_gravité
```

**Pression (forme symétrique, conserve la quantité de mouvement) :**
```
Fₚ = –Σⱼ mⱼ · (Pᵢ/ρᵢ² + Pⱼ/ρⱼ²) · ∇W_spiky
```

**Viscosité :**
```
Fᵥ = μ · Σⱼ mⱼ · (vⱼ–vᵢ)/ρⱼ · ∇²W_visc
```

### Intégration (Euler symplectique)

```
v(t+dt) = v(t) + (F/ρ) · dt
x(t+dt) = x(t) + v(t+dt) · dt
```

Les conditions aux limites sont **réflectives avec amortissement** (coefficient 0,4).

---

## Architecture du projet

```
gpu-fluids-simulation/
├── Package.swift
└── Sources/SPHFluid/
    ├── Types.swift          ← Structures partagées CPU/GPU (Particle, SimParams, CameraUniforms)
    ├── ShaderSources.swift  ← Code MSL embarqué comme String Swift
    │     ├── clearGrid              kernel — réinitialise les compteurs de cellules
    │     ├── buildGrid              kernel — insère chaque particule dans sa cellule
    │     ├── computeDensityPressure kernel — calcule ρ et P
    │     ├── computeForces          kernel — accumule pression + viscosité + gravité
    │     ├── integrate              kernel — Euler symplectique + frontières
    │     ├── particleVertex         vertex shader — billboard vers la caméra
    │     └── particleFragment       fragment shader — impostor sphérique éclairé
    ├── SPHSimulator.swift   ← Gère buffers GPU et enchaîne les 5 kernels compute
    ├── Camera.swift         ← Caméra orbite (azimut, élévation, zoom)
    ├── Renderer.swift       ← MTKViewDelegate — cadence simulation + draw call
    └── main.swift           ← Fenêtre AppKit, MTKView, boucle événements
```

### Pipeline par frame

```
┌─────────────────────────────────────────────────────────┐
│  Compute Command Encoder                                │
│  1. clearGrid              (1 thread / cellule)         │
│  2. buildGrid              (1 thread / particule)       │
│  3. computeDensityPressure (1 thread / particule)       │
│  4. computeForces          (1 thread / particule)       │
│  5. integrate              (1 thread / particule)       │
└─────────────────────────────────────────────────────────┘
          ↓  même command queue, exécution ordonnée
┌─────────────────────────────────────────────────────────┐
│  Render Command Encoder                                 │
│  drawPrimitives(triangleStrip, 4 verts, N instances)   │
│  → particleVertex  (billboard)                         │
│  → particleFragment (impostor + Phong)                  │
└─────────────────────────────────────────────────────────┘
```

### Grille uniforme (spatial hashing)

Chaque cellule de taille `h` stocke les indices des particules qu'elle contient.
La recherche de voisins parcourt les **27 cellules adjacentes** (3×3×3), ce qui réduit la complexité de O(N²) à O(N·k) où k est le nombre moyen de voisins.

```
gridCount[cellIdx]                          → nombre de particules dans la cellule
gridParticles[cellIdx * maxPerCell + slot]  → indice de la particule
```

### Sphere impostors

Au lieu de gérer un mesh 3D par particule :
- On dessine **un quad de 4 sommets** par particule (instancing).
- Le vertex shader oriente le quad vers la caméra (**billboarding**).
- Le fragment shader reconstruit la normale de sphère à partir des coordonnées UV du quad et **écarte** les fragments hors du cercle unité.
- L'éclairage Phong s'applique sur cette normale reconstruite.

Résultat : apparence de sphères lisses pour **4 vertex par particule** au lieu de ~100+.

---

## Prérequis

- macOS 13 Ventura ou plus récent
- Mac avec GPU compatible Metal (Intel, Apple Silicon, AMD discret)
- Swift 5.9+ (inclus dans Xcode 15+ ou `swift.org`)

Vérifier la version Swift :
```bash
swift --version
```

---

## Lancer le projet

```bash
cd gpu-fluids-simulation
swift run -c release
```

> Le flag `-c release` active les optimisations LLVM — indispensable pour tenir 60 fps avec 6 000 particules.

La compilation initiale du shader Metal a lieu au premier lancement (MTLDevice.makeLibrary), ce qui peut prendre 2–3 secondes.

---

## Contrôles

| Entrée | Action |
|--------|--------|
| Clic-glisser souris | Orbiter la caméra |
| Molette | Zoom avant / arrière |
| `R` | Réinitialiser la simulation |
| `←` / `→` | Réduire / augmenter les sous-pas par frame |
| `Q` ou `Échap` | Quitter |

---

## Paramètres à tweaker

Tous les paramètres physiques sont dans `SPHSimulator.swift`, méthode `init` :

| Paramètre | Valeur par défaut | Effet |
|-----------|------------------|-------|
| `numParticles` | 6 000 | Nombre de particules |
| `h` | 0.08 m | Rayon de lissage |
| `mass` | 0.02 kg | Masse par particule |
| `restDensity` | 1000 kg/m³ | Densité de repos |
| `gasConstant` | 200 | Raideur de pression |
| `viscosity` | 0.15 | Viscosité dynamique |
| `dt` | 0.003 s | Pas de temps |
| `gravity` | –9.81 m/s² | Gravité |

---

## Limitations connues et pistes d'amélioration

- **Surface reconstruction** : le rendu actuel est à base de particules. Une amélioration naturelle serait d'implémenter le **Marching Cubes** sur GPU pour extraire une iso-surface lisse.
- **Tri radix GPU** : remplacer le hash spatial par un tri des particules améliorerait la cohérence cache.
- **PCISPH / IISPH** : des solveurs de pression plus avancés permettraient un dt plus grand et moins d'oscillations.
- **Surface tension** : le terme de cohésion (Akinci 2013) rendrait les gouttelettes plus réalistes.
- **Rendu volumétrique** : screen-space fluid rendering (NVIDIA) pour un aspect liquide translucide.

---

## Références

- Müller et al. (2003) — *Particle-Based Fluid Simulation for Interactive Applications*
- Becker & Teschner (2007) — *Weakly Compressible SPH for Free Surface Flows*
- Green (2010) — *Particle Simulation using CUDA* (NVIDIA SDK)
- Lorensen & Cline (1987) — *Marching Cubes: A High Resolution 3D Surface Construction Algorithm*
