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
        // Prepare the bowGroup placeholder so any legacy reference to this.bowGroup doesn't crash
        this.bowGroup = new THREE.Group();
        this.scene.add(this.bowGroup);

        const loader = new THREE.FBXLoader();
        
        // Load Character
        loader.load('/static/models/X Bot.fbx', (object) => {
            this.archerGroup = object;
            
            object.scale.set(0.012, 0.012, 0.012);
            object.position.set(0, 0, 0);
            
            // Cast shadows
            object.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    if (child.material) {
                        // Ensure materials are somewhat reflective PBR if missing
                        child.material.roughness = 0.5;
                        child.material.metalness = 0.1;
                    }
                }
            });

            this.scene.add(object);

            // Setup Animations
            this.mixer = new THREE.AnimationMixer(object);
            this.actions = {};
            
            // Load Idle Animation
            loader.load('/static/models/standing idle 01.fbx', (anim) => {
                const idleAction = this.mixer.clipAction(anim.animations[0]);
                idleAction.play();
                this.actions['idle'] = idleAction;
                this.activeAction = idleAction;
            });
            
            // Load Firing Animations (Pre-load for later)
            loader.load('/static/models/standing draw arrow.fbx', (anim) => {
                const drawAction = this.mixer.clipAction(anim.animations[0]);
                drawAction.clampWhenFinished = true;
                this.actions['draw'] = drawAction;
            });
            
            loader.load('/static/models/standing aim overdraw.fbx', (anim) => {
                const overdrawAction = this.mixer.clipAction(anim.animations[0]);
                this.actions['overdraw'] = overdrawAction;
            });

        }, undefined, (error) => {
            console.error("Error loading FBX:", error);
        });
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

        if (this.actions && this.actions['draw']) {
            this.actions['draw'].reset()
                .setEffectiveWeight(1)
                .setEffectiveTimeScale(1.5)
                .crossFadeFrom(this.activeAction, 0.3)
                .play();
            this.activeAction = this.actions['draw'];
            
            setTimeout(() => {
                this._launchArrowMesh(trajectoryPoints, onComplete);
                
                if (this.actions['idle']) {
                    this.actions['idle'].reset()
                        .setEffectiveWeight(1)
                        .crossFadeFrom(this.activeAction, 0.5)
                        .play();
                    this.activeAction = this.actions['idle'];
                }
            }, 500);
        } else {
            this._launchArrowMesh(trajectoryPoints, onComplete);
        }
    }

    _launchArrowMesh(trajectoryPoints, onComplete) {
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

        if (this.mixer) {
            this.mixer.update(dt);
        }

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
