/**
 * LEAA Archery Demo — Three.js Scene (PBR / Cinematic)
 *
 * Realistic archery range with PBR materials, smooth shading,
 * ACES tone mapping, and anatomically detailed character.
 * Physics coords [x,y,z] are received as Three.js [x,z,y] from the backend.
 */

class ArcheryScene {
    constructor(container) {
        this.container = container;
        this.targets = {};
        this.targetMeshes = {};
        this.highlightRings = {};
        this.flags = {};
        this.arrowMesh = null;
        this.trailLine = null;
        this.particles = [];
        this.animatingArrow = false;
        this.clock = new THREE.Clock();

        this._initRenderer();
        this._initCamera();
        this._initLights();
        this._buildEnvironment();
        this._buildArcher();
        this.animate();
    }

    // -------------------------------------------------------------------
    // Renderer — PBR pipeline, ACES tone mapping, high-res shadows
    // -------------------------------------------------------------------

    _initRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            powerPreference: 'high-performance',
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.1;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.physicallyCorrectLights = true;
        this.container.appendChild(this.renderer.domElement);

        this.scene = new THREE.Scene();

        // Gradient sky via vertex colors
        this.scene.background = new THREE.Color(0x78b9e8);
        this.scene.fog = new THREE.FogExp2(0x9cc8e8, 0.008);

        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    // -------------------------------------------------------------------
    // Camera — cinematic FOV, fixed perspective
    // -------------------------------------------------------------------

    _initCamera() {
        this.camera = new THREE.PerspectiveCamera(
            45, window.innerWidth / window.innerHeight, 0.1, 300
        );
        this.camera.position.set(-4.85, 2.21, 0.41);
        this.camera.lookAt(15.00, 1.30, -0.60);
    }

    // -------------------------------------------------------------------
    // Lighting — 3-point + hemisphere for realistic PBR illumination
    // -------------------------------------------------------------------

    _initLights() {
        // Hemisphere for ambient sky/ground bounce
        const hemi = new THREE.HemisphereLight(0x8dc1f2, 0x3d6b2e, 0.6);
        this.scene.add(hemi);

        // Key light (warm sun)
        const sun = new THREE.DirectionalLight(0xfff0dd, 2.5);
        sun.position.set(25, 35, 15);
        sun.castShadow = true;
        sun.shadow.mapSize.set(2048, 2048);
        sun.shadow.camera.near = 0.5;
        sun.shadow.camera.far = 120;
        sun.shadow.camera.left = -60;
        sun.shadow.camera.right = 60;
        sun.shadow.camera.top = 40;
        sun.shadow.camera.bottom = -40;
        sun.shadow.bias = -0.0005;
        sun.shadow.normalBias = 0.02;
        this.scene.add(sun);

        // Fill light (cool, opposite side)
        const fill = new THREE.DirectionalLight(0x8aaed4, 0.8);
        fill.position.set(-20, 15, -10);
        this.scene.add(fill);

        // Rim/back light
        const rim = new THREE.DirectionalLight(0xffeebb, 0.5);
        rim.position.set(-10, 20, 25);
        this.scene.add(rim);
    }

    // -------------------------------------------------------------------
    // Environment — PBR ground, realistic trees, grass patches, sky dome
    // -------------------------------------------------------------------

    _buildEnvironment() {
        // Sky dome (gradient sphere)
        const skyGeo = new THREE.SphereGeometry(150, 32, 16);
        const skyMat = new THREE.ShaderMaterial({
            uniforms: {
                topColor: { value: new THREE.Color(0x4488cc) },
                bottomColor: { value: new THREE.Color(0xc8dff0) },
                offset: { value: 20 },
                exponent: { value: 0.5 },
            },
            vertexShader: `
                varying vec3 vWorldPosition;
                void main() {
                    vec4 worldPos = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPos.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 topColor;
                uniform vec3 bottomColor;
                uniform float offset;
                uniform float exponent;
                varying vec3 vWorldPosition;
                void main() {
                    float h = normalize(vWorldPosition + offset).y;
                    gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
                }
            `,
            side: THREE.BackSide,
        });
        this.scene.add(new THREE.Mesh(skyGeo, skyMat));

        // Ground — PBR with roughness
        const groundGeo = new THREE.PlaneGeometry(150, 100, 80, 60);
        groundGeo.rotateX(-Math.PI / 2);
        const gv = groundGeo.attributes.position;
        for (let i = 0; i < gv.count; i++) {
            const x = gv.getX(i), z = gv.getZ(i);
            gv.setY(i,
                Math.sin(x * 0.2) * 0.12 +
                Math.cos(z * 0.3) * 0.08 +
                Math.sin(x * 0.7 + z * 0.5) * 0.04 +
                (Math.random() - 0.5) * 0.03
            );
        }
        groundGeo.computeVertexNormals();
        const groundMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x4a8c3f,
            roughness: 0.9,
            metalness: 0.0,
        });
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Grass clumps (small displaced planes for texture)
        this._addGrassClumps();

        // Trees
        this._addTrees();

        // Fences
        this._addFences();

        // Distance markers
        this._addDistanceMarkers();

        // Rocks scattered around
        this._addRocks();
    }

    _addGrassClumps() {
        const grassMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x5ca84a,
            roughness: 0.85,
            metalness: 0,
            side: THREE.DoubleSide,
        });
        for (let i = 0; i < 200; i++) {
            const x = (Math.random() - 0.3) * 80;
            const z = (Math.random() - 0.5) * 50;
            const h = 0.15 + Math.random() * 0.2;
            const blade = new THREE.Mesh(
                new THREE.PlaneGeometry(0.04, h),
                grassMat
            );
            blade.position.set(x, h / 2, z);
            blade.rotation.y = Math.random() * Math.PI;
            blade.rotation.x = (Math.random() - 0.5) * 0.3;
            this.scene.add(blade);
        }
    }

    _addTrees() {
        const trunkMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x5c3a1e, roughness: 0.85, metalness: 0.0
        });
        const leafMats = [
            new THREE.MeshStandardMaterial({ flatShading: true, color: 0x2d7a2d, roughness: 0.7, metalness: 0 }),
            new THREE.MeshStandardMaterial({ flatShading: true, color: 0x1a5c1a, roughness: 0.75, metalness: 0 }),
            new THREE.MeshStandardMaterial({ flatShading: true, color: 0x3a9a3a, roughness: 0.65, metalness: 0 }),
        ];

        const treePositions = [
            [-8, 0, -15], [-5, 0, -18], [10, 0, -22], [25, 0, -24],
            [40, 0, -20], [55, 0, -22], [60, 0, -16],
            [-8, 0, 16], [-3, 0, 22], [15, 0, 24], [30, 0, 20],
            [45, 0, 22], [55, 0, 18], [65, 0, 14],
            [-12, 0, -8], [-12, 0, 8], [72, 0, -6], [72, 0, 6],
        ];

        treePositions.forEach(pos => {
            const group = new THREE.Group();
            const h = 3.0 + Math.random() * 3.5;

            // Trunk with slight taper
            const trunk = new THREE.Mesh(
                new THREE.CylinderGeometry(0.12, 0.2, h * 0.45, 8),
                trunkMat
            );
            trunk.position.y = h * 0.22;
            trunk.castShadow = true;
            trunk.receiveShadow = true;
            group.add(trunk);

            // Layered foliage spheres (more organic than cones)
            const mat = leafMats[Math.floor(Math.random() * leafMats.length)];
            for (let j = 0; j < 4; j++) {
                const r = (1.4 - j * 0.25) + Math.random() * 0.3;
                const foliage = new THREE.Mesh(
                    new THREE.SphereGeometry(r * 0.55, 8, 6),
                    mat
                );
                foliage.position.set(
                    (Math.random() - 0.5) * 0.3,
                    h * 0.4 + j * h * 0.15,
                    (Math.random() - 0.5) * 0.3
                );
                foliage.scale.y = 0.7 + Math.random() * 0.3;
                foliage.castShadow = true;
                group.add(foliage);
            }
            group.position.set(pos[0], pos[1], pos[2]);
            this.scene.add(group);
        });
    }

    _addFences() {
        const fenceMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x8b6914, roughness: 0.8, metalness: 0.05
        });
        for (let side = -1; side <= 1; side += 2) {
            for (let x = -5; x < 65; x += 3.5) {
                const post = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.05, 0.06, 1.3, 6),
                    fenceMat
                );
                post.position.set(x, 0.65, side * 13);
                post.castShadow = true;
                post.receiveShadow = true;
                this.scene.add(post);

                if (x < 62) {
                    for (let rh of [0.45, 0.9]) {
                        const rail = new THREE.Mesh(
                            new THREE.BoxGeometry(3.5, 0.07, 0.05),
                            fenceMat
                        );
                        rail.position.set(x + 1.75, rh, side * 13);
                        rail.castShadow = true;
                        this.scene.add(rail);
                    }
                }
            }
        }
    }

    _addDistanceMarkers() {
        const markerMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x888866, transparent: true, opacity: 0.4, roughness: 1
        });
        [10, 20, 30, 40, 50].forEach(d => {
            const marker = new THREE.Mesh(
                new THREE.PlaneGeometry(0.5, 2.5),
                markerMat
            );
            marker.rotation.x = -Math.PI / 2;
            marker.position.set(d, 0.02, 0);
            this.scene.add(marker);
        });
    }

    _addRocks() {
        const rockMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x777777, roughness: 0.9, metalness: 0.1
        });
        const rockPositions = [
            [5, 0, -6], [12, 0, 8], [35, 0, -10], [50, 0, 7],
            [-3, 0, -4], [60, 0, -3], [8, 0, 11], [42, 0, 9],
        ];
        rockPositions.forEach(pos => {
            const s = 0.15 + Math.random() * 0.35;
            const rock = new THREE.Mesh(
                new THREE.DodecahedronGeometry(s, 1),
                rockMat
            );
            rock.position.set(pos[0], s * 0.4, pos[2]);
            rock.rotation.set(Math.random(), Math.random(), Math.random());
            rock.scale.set(1, 0.5 + Math.random() * 0.5, 1);
            rock.castShadow = true;
            rock.receiveShadow = true;
            this.scene.add(rock);
        });
    }

    // -------------------------------------------------------------------
    // Archer — smooth PBR character with anatomical proportions
    // -------------------------------------------------------------------

    _buildArcher() {
        const group = new THREE.Group();
        const S = 6; // segment count for low-poly geometry

        // PBR Materials
        const skinMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0xffdcb3, roughness: 0.8, metalness: 0.1
        });
        const tunicMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0xf2f2f2, roughness: 0.9, metalness: 0.1
        });
        const leatherMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x222222, roughness: 0.6, metalness: 0.2
        });
        const bootMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0xffffff, roughness: 0.8, metalness: 0.1
        });
        const pantsMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0xf2f2f2, roughness: 0.9, metalness: 0.1
        });
        const hairMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x3d2314, roughness: 0.9, metalness: 0.1
        });
        const metalMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x999999, roughness: 0.3, metalness: 0.7
        });
        const goldMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0xcccccc, roughness: 0.3, metalness: 0.8
        });
        const woodMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x6b3a1f, roughness: 0.7, metalness: 0.05
        });

        // --- Boots ---
        for (let side = -1; side <= 1; side += 2) {
            // Boot shaft
            const bootShaft = new THREE.Mesh(
                new THREE.CylinderGeometry(0.065, 0.07, 0.25, S), bootMat
            );
            bootShaft.position.set(0, 0.14, side * 0.11);
            bootShaft.castShadow = true;
            group.add(bootShaft);

            // Boot toe
            const toe = new THREE.Mesh(
                new THREE.SphereGeometry(0.07, S, S / 2, 0, Math.PI * 2, 0, Math.PI / 2), bootMat
            );
            toe.rotation.x = Math.PI / 2;
            toe.position.set(0.06, 0.02, side * 0.11);
            group.add(toe);

            // Boot sole
            const sole = new THREE.Mesh(
                new THREE.BoxGeometry(0.16, 0.03, 0.14), bootMat
            );
            sole.position.set(0.02, 0.015, side * 0.11);
            group.add(sole);
        }

        // --- Legs ---
        for (let side = -1; side <= 1; side += 2) {
            // Shin
            const shin = new THREE.Mesh(
                new THREE.CylinderGeometry(0.05, 0.06, 0.38, S), pantsMat
            );
            shin.position.set(0, 0.46, side * 0.11);
            shin.castShadow = true;
            group.add(shin);

            // Knee joint
            const knee = new THREE.Mesh(
                new THREE.SphereGeometry(0.055, S, S), pantsMat
            );
            knee.position.set(0, 0.63, side * 0.11);
            group.add(knee);

            // Thigh
            const thigh = new THREE.Mesh(
                new THREE.CylinderGeometry(0.065, 0.055, 0.32, S), pantsMat
            );
            thigh.position.set(0, 0.8, side * 0.11);
            thigh.castShadow = true;
            group.add(thigh);
        }

        // --- Hips ---
        const hips = new THREE.Mesh(
            new THREE.SphereGeometry(0.2, S, S), pantsMat
        );
        hips.scale.set(1, 0.55, 0.85);
        hips.position.set(0, 0.92, 0);
        group.add(hips);

        // --- Torso (built from multiple smooth shapes) ---
        // Lower torso / waist
        
        const shirtMat = new THREE.MeshStandardMaterial({ flatShading: true, color: 0xe86a33, roughness: 0.9, metalness: 0.1 });
        const waist = new THREE.Mesh(
            new THREE.CylinderGeometry(0.17, 0.2, 0.25, S), shirtMat
        );
        waist.position.set(0, 1.08, 0);
        waist.castShadow = true;
        group.add(waist);

        // Mid torso
        const midTorso = new THREE.Mesh(
            new THREE.CylinderGeometry(0.2, 0.17, 0.3, S), tunicMat
        );
        midTorso.position.set(0, 1.3, 0);
        midTorso.castShadow = true;
        group.add(midTorso);

        // Chest
        const chest = new THREE.Mesh(
            new THREE.SphereGeometry(0.21, S, S), tunicMat
        );
        chest.scale.set(1, 0.75, 0.85);
        chest.position.set(0, 1.48, 0);
        chest.castShadow = true;
        group.add(chest);

        // --- Belt ---
        const belt = new THREE.Mesh(
            new THREE.TorusGeometry(0.19, 0.025, S, S), leatherMat
        );
        belt.rotation.x = Math.PI / 2;
        belt.position.set(0, 0.95, 0);
        group.add(belt);

        // Belt buckle
        const buckle = new THREE.Mesh(
            new THREE.BoxGeometry(0.05, 0.05, 0.015), goldMat
        );
        buckle.position.set(0.2, 0.95, 0);
        group.add(buckle);

        // --- Neck ---
        const neck = new THREE.Mesh(
            new THREE.CylinderGeometry(0.055, 0.065, 0.1, S), skinMat
        );
        neck.position.set(0, 1.65, 0);
        group.add(neck);

        // --- Head ---
        const head = new THREE.Mesh(
            new THREE.SphereGeometry(0.14, S * 2, S), skinMat
        );
        head.scale.set(1, 1.08, 0.95);
        head.position.set(0, 1.82, 0);
        head.castShadow = true;
        group.add(head);

        // Jaw
        const jaw = new THREE.Mesh(
            new THREE.SphereGeometry(0.1, S, S, 0, Math.PI * 2, Math.PI * 0.4, Math.PI * 0.3), skinMat
        );
        jaw.position.set(0.02, 1.75, 0);
        group.add(jaw);

        // Ears
        for (let side = -1; side <= 1; side += 2) {
            const ear = new THREE.Mesh(
                new THREE.SphereGeometry(0.03, 8, 6), skinMat
            );
            ear.scale.set(0.5, 1, 0.7);
            ear.position.set(-0.01, 1.82, side * 0.14);
            group.add(ear);
        }

        // Nose
        const nose = new THREE.Mesh(
            new THREE.ConeGeometry(0.02, 0.04, 6), skinMat
        );
        nose.rotation.x = -Math.PI / 2;
        nose.position.set(0.15, 1.81, 0);
        group.add(nose);

        // Eyes
        for (let side = -1; side <= 1; side += 2) {
            // Eye white
            const eyeWhite = new THREE.Mesh(
                new THREE.SphereGeometry(0.02, 8, 6),
                new THREE.MeshStandardMaterial({ flatShading: true, color: 0xf5f5f0, roughness: 0.3 })
            );
            eyeWhite.position.set(0.125, 1.84, side * 0.05);
            group.add(eyeWhite);
            // Iris
            const iris = new THREE.Mesh(
                new THREE.SphereGeometry(0.012, 8, 6),
                new THREE.MeshStandardMaterial({ flatShading: true, color: 0x3a6b3a, roughness: 0.4 })
            );
            iris.position.set(0.14, 1.84, side * 0.05);
            group.add(iris);
            // Pupil
            const pupil = new THREE.Mesh(
                new THREE.SphereGeometry(0.006, 6, 4),
                new THREE.MeshBasicMaterial({ flatShading: true, color: 0x111111 })
            );
            pupil.position.set(0.148, 1.84, side * 0.05);
            group.add(pupil);

            // Eyebrow
            const brow = new THREE.Mesh(
                new THREE.BoxGeometry(0.04, 0.008, 0.008), hairMat
            );
            brow.position.set(0.13, 1.87, side * 0.05);
            brow.rotation.z = side * 0.15;
            group.add(brow);
        }

        // Hair
        const hair = new THREE.Mesh(
            new THREE.SphereGeometry(0.155, S, S, 0, Math.PI * 2, 0, Math.PI * 0.55), hairMat
        );
        hair.position.set(-0.01, 1.85, 0);
        group.add(hair);

        // Hair band
        const hairBand = new THREE.Mesh(
            new THREE.TorusGeometry(0.145, 0.01, 8, S), leatherMat
        );
        hairBand.rotation.x = Math.PI / 2;
        hairBand.rotation.z = 0.15;
        hairBand.position.set(0, 1.87, 0);
        group.add(hairBand);

        // --- Shoulders ---
        for (let side = -1; side <= 1; side += 2) {
            const shoulder = new THREE.Mesh(
                new THREE.SphereGeometry(0.065, S, S), tunicMat
            );
            shoulder.position.set(0, 1.52, side * 0.24);
            shoulder.castShadow = true;
            group.add(shoulder);

            // Shoulder pad
            const pad = new THREE.Mesh(
                new THREE.SphereGeometry(0.07, S, S / 2, 0, Math.PI * 2, 0, Math.PI / 2), leatherMat
            );
            pad.position.set(0, 1.55, side * 0.25);
            group.add(pad);
        }

        // --- Left arm (bow arm — extended) ---
        const leftArm = new THREE.Group();
        const lUpper = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.045, 0.3, S), tunicMat);
        lUpper.rotation.z = Math.PI / 2.2;
        lUpper.position.set(0.14, 0, 0);
        lUpper.castShadow = true;
        leftArm.add(lUpper);

        const lElbow = new THREE.Mesh(new THREE.SphereGeometry(0.04, S, S), skinMat);
        lElbow.position.set(0.28, 0.02, 0);
        leftArm.add(lElbow);

        const lFore = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.04, 0.28, S), skinMat);
        lFore.rotation.z = Math.PI / 2;
        lFore.position.set(0.42, 0.02, 0);
        lFore.castShadow = true;
        leftArm.add(lFore);

        // Bracer
        const bracer = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.04, 0.1, S), leatherMat);
        bracer.rotation.z = Math.PI / 2;
        bracer.position.set(0.36, 0.02, 0);
        leftArm.add(bracer);

        // Hand
        const lHand = new THREE.Mesh(new THREE.SphereGeometry(0.035, S, S), skinMat);
        lHand.scale.set(0.8, 1, 0.7);
        lHand.position.set(0.56, 0.02, 0);
        leftArm.add(lHand);

        leftArm.position.set(0, 1.47, -0.24);
        group.add(leftArm);

        // --- Right arm (draw arm — pulled back) ---
        const rightArm = new THREE.Group();
        const rUpper = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.045, 0.3, S), tunicMat);
        rUpper.rotation.z = -Math.PI / 3;
        rUpper.position.set(-0.06, -0.06, 0);
        rUpper.castShadow = true;
        rightArm.add(rUpper);

        const rElbow = new THREE.Mesh(new THREE.SphereGeometry(0.04, S, S), skinMat);
        rElbow.position.set(-0.12, -0.18, 0);
        rightArm.add(rElbow);

        const rFore = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.04, 0.25, S), skinMat);
        rFore.rotation.z = -Math.PI / 1.6;
        rFore.position.set(-0.1, -0.28, 0);
        rFore.castShadow = true;
        rightArm.add(rFore);

        // Draw hand with finger tab
        const rHand = new THREE.Mesh(new THREE.SphereGeometry(0.035, S, S), skinMat);
        rHand.scale.set(0.8, 1, 0.7);
        rHand.position.set(-0.04, -0.36, 0);
        rightArm.add(rHand);

        const fTab = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.035, 0.05), leatherMat);
        fTab.position.set(-0.04, -0.36, 0);
        rightArm.add(fTab);

        rightArm.position.set(0, 1.52, 0.24);
        group.add(rightArm);

        // --- Quiver ---
        const quiver = new THREE.Mesh(
            new THREE.CylinderGeometry(0.05, 0.04, 0.5, S), leatherMat
        );
        quiver.rotation.z = 0.2;
        quiver.position.set(-0.18, 1.22, 0.06);
        quiver.castShadow = true;
        group.add(quiver);

        // Quiver strap
        const strapCurve = new THREE.QuadraticBezierCurve3(
            new THREE.Vector3(-0.18, 1.5, 0.06),
            new THREE.Vector3(0.1, 1.5, 0.15),
            new THREE.Vector3(0.05, 1.35, -0.1)
        );
        const strap = new THREE.Mesh(
            new THREE.TubeGeometry(strapCurve, 12, 0.012, 6, false), leatherMat
        );
        group.add(strap);

        // Arrows in quiver
        const arrowShaftMat = new THREE.MeshStandardMaterial({ flatShading: true, color: 0xc8a85c, roughness: 0.6 });
        const fletchColors = [0xcc3333, 0x3355cc, 0xcc3333, 0x33aa44, 0xccaa22];
        for (let i = 0; i < 5; i++) {
            const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.22, 6), arrowShaftMat);
            shaft.position.set(-0.18 + (i - 2) * 0.015, 1.55, 0.06 + (i % 2) * 0.015);
            shaft.rotation.z = 0.12 + i * 0.02;
            group.add(shaft);

            const fl = new THREE.Mesh(
                new THREE.PlaneGeometry(0.025, 0.035),
                new THREE.MeshStandardMaterial({ flatShading: true, color: fletchColors[i], side: THREE.DoubleSide, roughness: 0.5 })
            );
            fl.position.set(-0.18 + (i - 2) * 0.015, 1.64, 0.06 + (i % 2) * 0.015);
            group.add(fl);
        }

        // --- Bow ---
        const bowGroup = new THREE.Group();

        // Bow limbs (tube geometry for smooth curve)
        const bowCurve = new THREE.QuadraticBezierCurve3(
            new THREE.Vector3(0, -0.48, 0),
            new THREE.Vector3(0.24, 0, 0),
            new THREE.Vector3(0, 0.48, 0)
        );
        const bowTube = new THREE.TubeGeometry(bowCurve, 24, 0.016, S, false);
        const bowMesh = new THREE.Mesh(bowTube, woodMat);
        bowMesh.castShadow = true;
        bowGroup.add(bowMesh);

        // Decorative wrap at grip
        const gripWrap = new THREE.Mesh(
            new THREE.TorusGeometry(0.02, 0.008, 8, S), leatherMat
        );
        gripWrap.rotation.y = Math.PI / 2;
        gripWrap.position.set(0.22, 0, 0);
        bowGroup.add(gripWrap);

        // Grip
        const grip = new THREE.Mesh(
            new THREE.CylinderGeometry(0.022, 0.022, 0.09, S), leatherMat
        );
        grip.position.set(0.2, 0, 0);
        bowGroup.add(grip);

        // Bow tips with metal nocks
        for (let end = -1; end <= 1; end += 2) {
            const nock = new THREE.Mesh(new THREE.SphereGeometry(0.01, 8, 6), metalMat);
            nock.position.set(0, end * 0.48, 0);
            bowGroup.add(nock);
        }

        // Bowstring
        const stringPts = [
            new THREE.Vector3(0, -0.48, 0),
            new THREE.Vector3(-0.18, 0, 0),
            new THREE.Vector3(0, 0.48, 0),
        ];
        const sCurve = new THREE.QuadraticBezierCurve3(...stringPts);
        const sGeo = new THREE.BufferGeometry().setFromPoints(sCurve.getPoints(24));
        bowGroup.add(new THREE.Line(sGeo, new THREE.LineBasicMaterial({ color: 0xeeeedd })));

        // Nocked arrow
        const nocked = new THREE.Group();
        const ns = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.6, 6), arrowShaftMat);
        ns.rotation.z = Math.PI / 2;
        nocked.add(ns);
        const nt = new THREE.Mesh(new THREE.ConeGeometry(0.018, 0.05, 6), metalMat);
        nt.rotation.z = -Math.PI / 2;
        nt.position.x = 0.33;
        nocked.add(nt);
        for (let f = 0; f < 3; f++) {
            const flt = new THREE.Mesh(
                new THREE.PlaneGeometry(0.05, 0.025),
                new THREE.MeshStandardMaterial({ flatShading: true, color: 0xcc3333, side: THREE.DoubleSide, roughness: 0.5 })
            );
            flt.position.x = -0.26;
            flt.rotation.x = (f / 3) * Math.PI * 2;
            nocked.add(flt);
        }
        nocked.position.set(-0.08, 0, 0);
        bowGroup.add(nocked);

        bowGroup.position.set(0.56, 1.49, -0.24);
        this.bowGroup = bowGroup;
        group.add(bowGroup);

        // --- Cape ---
        const capeGeo = new THREE.PlaneGeometry(0.45, 0.65, 4, 6);
        const cv = capeGeo.attributes.position;
        for (let i = 0; i < cv.count; i++) {
            const y = cv.getY(i);
            cv.setZ(i, -0.04 - Math.abs(y) * 0.12 + Math.sin(cv.getX(i) * 4) * 0.02);
        }
        capeGeo.computeVertexNormals();
        const capeMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x1a3f1a, roughness: 0.7, metalness: 0, side: THREE.DoubleSide
        });
        const cape = new THREE.Mesh(capeGeo, capeMat);
        cape.visible = false;
        cape.position.set(-0.16, 1.32, 0);
        cape.rotation.y = Math.PI / 2;
        // cape.castShadow = true;
        this.cape = cape;
        group.add(cape);

        // Cape clasp
        const clasp = new THREE.Mesh(new THREE.SphereGeometry(0.015, 8, 6), goldMat);
        clasp.position.set(-0.05, 1.58, 0.12);
        group.add(clasp);
        const clasp2 = new THREE.Mesh(new THREE.SphereGeometry(0.015, 8, 6), goldMat);
        clasp2.position.set(-0.05, 1.58, -0.12);
        group.add(clasp2);

        group.position.set(0, 0, 0);
        this.archerGroup = group;
        this.scene.add(group);
    }

    // -------------------------------------------------------------------
    // Targets
    // -------------------------------------------------------------------

    buildTargets(sceneData) {
        Object.values(this.targetMeshes).forEach(g => this.scene.remove(g));
        Object.values(this.highlightRings).forEach(r => this.scene.remove(r));
        this.targetMeshes = {};
        this.highlightRings = {};
        this.targets = {};
        this.flags = {};

        sceneData.objects.forEach(obj => {
            this.targets[obj.id] = obj;
            const group = this._createTargetMesh(obj);
            group.position.set(obj.position[0], obj.position[1], obj.position[2]);
            this.scene.add(group);
            this.targetMeshes[obj.id] = group;

            const ringGeo = new THREE.RingGeometry(0.7, 0.85, 32);
            const ringMat = new THREE.MeshBasicMaterial({ flatShading: true,
                color: 0xffff00, side: THREE.DoubleSide, transparent: true, opacity: 0,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.set(obj.position[0], 0.05, obj.position[2]);
            this.scene.add(ring);
            this.highlightRings[obj.id] = ring;
        });
    }

    _createTargetMesh(obj) {
        const group = new THREE.Group();
        const S = 16;
        const colorMap = {
            red: 0xcc2222, blue: 0x2244cc, yellow: 0xccaa11,
            green: 0x22aa33, white: 0xdddddd,
        };
        const mainColor = colorMap[obj.flag_color] || 0xcc2222;
        const woodMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x7a5520, roughness: 0.75, metalness: 0.05
        });

        // Tripod stand
        for (let a = -1; a <= 1; a++) {
            const leg = new THREE.Mesh(
                new THREE.CylinderGeometry(0.025, 0.03, 2.0, 6), woodMat
            );
            leg.position.set(a * 0.15, -0.1, a === 0 ? 0.2 : -0.1);
            leg.rotation.z = a * 0.08;
            leg.rotation.x = (a === 0 ? -0.1 : 0.05);
            leg.castShadow = true;
            group.add(leg);
        }

        // Cross brace
        const brace = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.03, 0.03), woodMat);
        brace.position.set(0, 0.6, 0);
        group.add(brace);

        // Target face — thick disc + concentric painted rings
        const faceGeo = new THREE.CylinderGeometry(0.5, 0.5, 0.06, 32);
        const faceMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0xf5e8c0, roughness: 0.6, metalness: 0
        });
        const face = new THREE.Mesh(faceGeo, faceMat);
        face.rotation.z = Math.PI / 2;
        face.position.set(-0.04, 0, 0);
        face.castShadow = true;
        face.receiveShadow = true;
        group.add(face);

        // Painted rings on face
        const rings = [
            { radius: 0.48, color: 0xffffff },
            { radius: 0.38, color: 0x111111 },
            { radius: 0.28, color: 0x2277cc },
            { radius: 0.2, color: mainColor },
            { radius: 0.1, color: 0xffcc00 },
        ];
        rings.forEach((r, i) => {
            const geo = new THREE.CircleGeometry(r.radius, 32);
            const mat = new THREE.MeshStandardMaterial({ flatShading: true,
                color: r.color, side: THREE.DoubleSide, roughness: 0.5
            });
            const circle = new THREE.Mesh(geo, mat);
            circle.rotation.y = Math.PI / 2;
            circle.position.set(-0.075 - i * 0.001, 0, 0);
            group.add(circle);
        });

        // Flag on top
        const flagPole = new THREE.Mesh(
            new THREE.CylinderGeometry(0.012, 0.012, 0.55, 6), woodMat
        );
        flagPole.position.y = 1.25;
        group.add(flagPole);

        const flagGeo = new THREE.PlaneGeometry(0.3, 0.18, 4, 2);
        const fv = flagGeo.attributes.position;
        for (let i = 0; i < fv.count; i++) {
            fv.setZ(i, Math.sin(fv.getX(i) * 8) * 0.02);
        }
        flagGeo.computeVertexNormals();
        const flagMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: mainColor, side: THREE.DoubleSide, roughness: 0.5
        });
        const flag = new THREE.Mesh(flagGeo, flagMat);
        flag.position.set(0.16, 1.44, 0);
        group.add(flag);
        this.flags[obj.id] = flag;

        return group;
    }

    highlightTarget(targetId) {
        Object.values(this.highlightRings).forEach(r => { r.material.opacity = 0; });
        const ring = this.highlightRings[targetId];
        if (ring) {
            ring.material.opacity = 0.7;
            ring._pulseTime = 0;
        }
    }

    // -------------------------------------------------------------------
    // Arrow (fired projectile)
    // -------------------------------------------------------------------

    _createArrow() {
        const group = new THREE.Group();
        const arrowMat = new THREE.MeshStandardMaterial({ flatShading: true, color: 0xc8a85c, roughness: 0.5 });
        const metalMat = new THREE.MeshStandardMaterial({ flatShading: true, color: 0x888888, roughness: 0.3, metalness: 0.6 });

        const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.65, 8), arrowMat);
        shaft.rotation.z = Math.PI / 2;
        group.add(shaft);

        const tip = new THREE.Mesh(new THREE.ConeGeometry(0.025, 0.08, 6), metalMat);
        tip.rotation.z = -Math.PI / 2;
        tip.position.x = 0.37;
        group.add(tip);

        const fletchMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0xcc3333, side: THREE.DoubleSide, roughness: 0.5
        });
        for (let i = 0; i < 3; i++) {
            const fletch = new THREE.Mesh(new THREE.PlaneGeometry(0.1, 0.04), fletchMat);
            fletch.position.x = -0.28;
            fletch.rotation.x = (i / 3) * Math.PI * 2;
            group.add(fletch);
        }
        return group;
    }

    fireArrow(trajectoryPoints, onComplete) {
        if (this.animatingArrow) return;
        this.animatingArrow = true;

        if (this.arrowMesh) this.scene.remove(this.arrowMesh);
        if (this.trailLine) this.scene.remove(this.trailLine);
        this.particles.forEach(p => this.scene.remove(p));
        this.particles = [];

        this.arrowMesh = this._createArrow();
        this.arrowMesh.position.set(
            trajectoryPoints[0][0], trajectoryPoints[0][1], trajectoryPoints[0][2]
        );
        this.scene.add(this.arrowMesh);

        const trailPositions = [];
        const trailGeo = new THREE.BufferGeometry();
        const trailMat = new THREE.LineBasicMaterial({
            color: 0xffaa44, transparent: true, opacity: 0.6,
        });
        this.trailLine = new THREE.Line(trailGeo, trailMat);
        this.scene.add(this.trailLine);

        const totalPoints = trajectoryPoints.length;
        const duration = Math.min(totalPoints * 12, 2500);
        let startTime = null;
        let currentIndex = 0;

        const animateStep = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            const progress = Math.min(elapsed / duration, 1.0);
            const targetIndex = Math.min(Math.floor(progress * totalPoints), totalPoints - 1);

            while (currentIndex <= targetIndex && currentIndex < totalPoints) {
                const pt = trajectoryPoints[currentIndex];
                trailPositions.push(pt[0], pt[1], pt[2]);
                currentIndex++;
            }

            const pt = trajectoryPoints[targetIndex];
            this.arrowMesh.position.set(pt[0], pt[1], pt[2]);

            if (targetIndex < totalPoints - 1) {
                const next = trajectoryPoints[Math.min(targetIndex + 1, totalPoints - 1)];
                const dir = new THREE.Vector3(
                    next[0] - pt[0], next[1] - pt[1], next[2] - pt[2]
                ).normalize();
                if (dir.length() > 0.001) {
                    const axis = new THREE.Vector3(1, 0, 0);
                    this.arrowMesh.quaternion.copy(
                        new THREE.Quaternion().setFromUnitVectors(axis, dir)
                    );
                }
            }

            trailGeo.setAttribute('position',
                new THREE.Float32BufferAttribute(trailPositions, 3));

            if (progress < 1.0) {
                requestAnimationFrame(animateStep);
            } else {
                this._spawnImpactParticles(pt);
                this.animatingArrow = false;
                if (onComplete) onComplete();
            }
        };
        requestAnimationFrame(animateStep);
    }

    _spawnImpactParticles(position) {
        for (let i = 0; i < 15; i++) {
            const geo = new THREE.SphereGeometry(0.02 + Math.random() * 0.04, 6, 4);
            const mat = new THREE.MeshBasicMaterial({ flatShading: true,
                color: new THREE.Color().setHSL(0.06 + Math.random() * 0.08, 0.7, 0.55),
                transparent: true, opacity: 1,
            });
            const p = new THREE.Mesh(geo, mat);
            p.position.set(position[0], position[1], position[2]);
            p._vel = new THREE.Vector3(
                (Math.random() - 0.5) * 4,
                Math.random() * 4 + 1,
                (Math.random() - 0.5) * 4
            );
            p._life = 1.0;
            this.scene.add(p);
            this.particles.push(p);
        }
    }

    // -------------------------------------------------------------------
    // Animation loop
    // -------------------------------------------------------------------

    animate() {
        requestAnimationFrame(() => this.animate());
        const dt = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        // Moving targets
        Object.keys(this.targets).forEach(id => {
            const obj = this.targets[id];
            if (obj.is_moving && this.targetMeshes[id]) {
                obj.position[0] += obj.velocity[0] * dt;
                obj.position[1] += obj.velocity[1] * dt;
                obj.position[2] += obj.velocity[2] * dt;
                if (Math.abs(obj.position[2]) > 10) {
                    obj.velocity[2] *= -1;
                    obj.position[2] = Math.sign(obj.position[2]) * 10;
                }
                this.targetMeshes[id].position.set(
                    obj.position[0], obj.position[1], obj.position[2]
                );
                if (this.highlightRings[id]) {
                    this.highlightRings[id].position.set(
                        obj.position[0], 0.05, obj.position[2]
                    );
                }
            }
        });

        // Flags
        Object.values(this.flags).forEach(flag => {
            if (flag) {
                flag.rotation.y = Math.sin(time * 3 + flag.id) * 0.15;
                flag.position.x = 0.16 + Math.sin(time * 4) * 0.02;
            }
        });

        // Archer idle
        if (this.archerGroup) {
            this.archerGroup.scale.y = 1 + Math.sin(time * 1.8) * 0.006;
            this.archerGroup.rotation.z = Math.sin(time * 0.7) * 0.003;
        }
        if (this.cape) {
            this.cape.rotation.x = Math.sin(time * 2.0) * 0.05;
            this.cape.rotation.z = Math.sin(time * 1.3 + 0.5) * 0.03;
        }

        // Highlight rings
        Object.values(this.highlightRings).forEach(ring => {
            if (ring.material.opacity > 0) {
                ring._pulseTime = (ring._pulseTime || 0) + dt;
                ring.material.opacity = 0.4 + Math.sin(ring._pulseTime * 4) * 0.3;
                ring.scale.setScalar(1 + Math.sin(ring._pulseTime * 3) * 0.1);
            }
        });

        // Particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p._life -= dt * 1.5;
            if (p._life <= 0) {
                this.scene.remove(p);
                this.particles.splice(i, 1);
                continue;
            }
            p.position.add(p._vel.clone().multiplyScalar(dt));
            p._vel.y -= 9.8 * dt;
            p.material.opacity = p._life;
            p.scale.setScalar(p._life);
        }

        this.renderer.render(this.scene, this.camera);
    }
}
