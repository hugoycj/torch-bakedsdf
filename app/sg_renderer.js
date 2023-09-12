/**
 * Different display modes for debugging rendering.
 * @enum {number}
 */
const DisplayModeType = {
    /** Runs the full model with view dependence. */
    DISPLAY_FULL: 0,
    /** Disables the view-dependence network. */
    DISPLAY_DIFFUSE: 1,
    /** Only shows the view dependent component. */
    DISPLAY_VIEW_DEPENDENT: 2 ,
    /** Visualizes the surface normals of the mesh. */
    DISPLAY_NORMALS: 3 ,
    /** Visualizes the mesh using diffuse shading and a white albedo. */
    DISPLAY_SHADED: 4 ,
};

/**  @type {!DisplayModeType}  */
let gDisplayMode = DisplayModeType.DISPLAY_FULL;

/**
 * Creates a DOM element that belongs to the given CSS class.
 * @param {string} what
 * @param {string} classname
 * @return {!HTMLElement}
 */
function create(what, classname) {
    const e = /** @type {!HTMLElement} */(document.createElement(what));
    if (classname) {
        e.className = classname;
    }
    return e;
}

/**
 * Reports an error to the user by populating the error div with text.
 * @param {string} text
 */
 function error(text) {
    const e = document.getElementById('Loading');
    e.textContent = "Error: " + text;
    e.style.display = 'block';
    hideLoadingIndicator();
  }

/**
 * Resizes a DOM element to the given dimensions.
 * @param {!Element} element
 * @param {number} width
 * @param {number} height
 */
function setDims(element, width, height) {
    element.style.width = width.toFixed(2) + 'px';
    element.style.height = height.toFixed(2) + 'px';
}

/**
 * Hides the Loading indicator.
 */
function hideLoadingIndicator() {
    let loadingContainer = document.getElementById('loading-container');
    loadingContainer.style.display = 'none';
}

/**
 * Hides the Loading prompt and indicator.
 */
function hideLoading() {
    hideLoadingIndicator();
    let loading = document.getElementById('Loading');
    loading.style.display = 'none';
}


/**
 * The vertex shader to render our representation, this only interpolates the
 * vertex attributes and passes them onto the fragment shader.
 * @const {string}
 */
 const sgVertexShaderSource = `
 attribute vec3 _byte_normal;
 
 attribute vec3 _sg_mean_0;
 attribute vec3 _sg_mean_1;
 attribute vec3 _sg_mean_2;

 attribute float _sg_scale_0;
 attribute float _sg_scale_1;
 attribute float _sg_scale_2;

 attribute vec3 _sg_color_0;
 attribute vec3 _sg_color_1;
 attribute vec3 _sg_color_2;

 varying vec3 vDiffuse;
 varying vec3 vDirection;
 varying vec3 vPositionWorld;

 varying vec3 vSgMean0;
 varying vec3 vSgMean1;
 varying vec3 vSgMean2;

 varying float vSgScale0;
 varying float vSgScale1;
 varying float vSgScale2;

 varying vec3 vSgColor0;
 varying vec3 vSgColor1;
 varying vec3 vSgColor2;

 uniform int displayMode;
 uniform mat3 worldspace_R_opengl;
 uniform mat4 world_T_clip;
 
  void main() {
    // See the DisplayMode enum at the top of this file.
    // Runs the full model with view dependence.
    const int DISPLAY_FULL = 0;
    // Disables the view-dependence network.
    const int DISPLAY_DIFFUSE = 1;
    // Only shows the view dependent component.
    const int DISPLAY_VIEW_DEPENDENT = 2;
    // Visualizes the surface normals of the mesh.
    const int DISPLAY_NORMALS = 3;
    // Visualizes the mesh using diffuse shading and a white albedo.
    const int DISPLAY_SHADED = 4;

    vec3 positionWorld = position;
    vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(positionWorld, 1.0);
    gl_Position = positionClip;
    positionClip /= positionClip.w;

    vec4 nearPoint = world_T_clip * vec4(positionClip.x, positionClip.y, -1.0, 1.0);
    vec4 farPoint = world_T_clip * vec4(positionClip.x, positionClip.y, 1.0, 1.0);
    vec3 origin = nearPoint.xyz / nearPoint.w;
    vDirection = worldspace_R_opengl * normalize(farPoint.xyz / farPoint.w - origin);

    vDiffuse = color.rgb;
    if (displayMode == DISPLAY_NORMALS ||
        displayMode == DISPLAY_SHADED) {
        vPositionWorld = worldspace_R_opengl * position;
    }

    if (displayMode == DISPLAY_FULL || 
        displayMode == DISPLAY_VIEW_DEPENDENT) {
        vSgMean0 = _sg_mean_0 * (2.0 / 255.0) - 1.0;
        vSgMean1 = _sg_mean_1 * (2.0 / 255.0) - 1.0;
        vSgMean2 = _sg_mean_2 * (2.0 / 255.0) - 1.0;

        vSgScale0 = 100.0 * _sg_scale_0 / 255.0;
        vSgScale1 = 100.0 * _sg_scale_1 / 255.0;
        vSgScale2 = 100.0 * _sg_scale_2 / 255.0;

        vSgColor0 = _sg_color_0 / 255.0;
        vSgColor1 = _sg_color_1 / 255.0;
        vSgColor2 = _sg_color_2 / 255.0;
    }
 }
`;

/**
* This fragment shader evaluates view depdence using Spherical Gaussians parameterized
* by the reflected view direction.
* @const {string}
*/
const sgFragmentShaderSource = `
 uniform int displayMode;

 varying vec3 vDiffuse;
 varying vec3 vDirection;
 varying vec3 vPositionWorld;

 varying vec3 vSgMean0;
 varying vec3 vSgMean1;
 varying vec3 vSgMean2;

 varying float vSgScale0;
 varying float vSgScale1;
 varying float vSgScale2;

 varying vec3 vSgColor0;
 varying vec3 vSgColor1;
 varying vec3 vSgColor2;

 vec3 evalSphericalGaussian(vec3 direction, vec3 mean, float scale, vec3 color) {
    return color * exp(scale * (dot(direction, mean) - 1.0));
 }

vec3 compute_sh_shading(vec3 n) {
    // SH coefficients for the "Eucalyptus Grove" scene from 
    // "An Efficient Representation for Irradiance Environment Maps" 
    // [Ravamoorthi & Hanrahan, 2001]
    vec3 c[9] = vec3[](
        vec3(0.38, 0.43, 0.45),
        vec3(0.29, 0.36, 0.41),
        vec3(0.04, 0.03, 0.01),
        vec3(-0.10, -0.10, -0.09),
        vec3(-0.06, -0.06, -0.04),
        vec3(0.01, -0.01, -0.05),
        vec3(-0.09, -0.13, -0.15),
        vec3(-0.06, -0.05, -0.04),
        vec3(0.02, -0.00, -0.05)
    );

    // From the SH shading implementation in three js:
    // https://github.com/mrdoob/three.js/blob/master/src/math/SphericalHarmonics3.js
    vec3 color = c[0] * 0.282095;

    color += c[1] * 0.488603 * n.y;
    color += c[2] * 0.488603 * n.z;
    color += c[3] * 0.488603 * n.x;
    
    color += c[4] * 1.092548 * (n.x * n.y);
    color += c[5] * 1.092548 * (n.y * n.z);
    color += c[7] * 1.092548 * (n.x * n.z);
    color += c[6] * 0.315392 * (3.0 * n.z * n.z - 1.0);
    color += c[8] * 0.546274 * (n.x * n.x - n.y * n.y);

    // Brighten everything up a bit with and-tuned constants.
    return 1.66 * color + vec3(0.1, 0.1, 0.1);
}

 void main() {
    // See the DisplayMode enum at the top of this file.
    // Runs the full model with view dependence.
    const int DISPLAY_FULL = 0;
    // Disables the view-dependence network.
    const int DISPLAY_DIFFUSE = 1;
    // Only shows the view dependent component.
    const int DISPLAY_VIEW_DEPENDENT = 2;
    // Visualizes the surface normals of the mesh.
    const int DISPLAY_NORMALS = 3;
    // Visualizes the mesh using diffuse shading and a white albedo.
    const int DISPLAY_SHADED = 4;

    vec3 diffuse = vDiffuse;
    vec3 directionWorld = -normalize(vDirection);
    vec3 normal = normalize(cross(dFdx(vPositionWorld), dFdy(vPositionWorld)));

    vec3 viewDependence = evalSphericalGaussian(
        directionWorld, normalize(vSgMean0), vSgScale0, vSgColor0);
    viewDependence += evalSphericalGaussian(
        directionWorld, normalize(vSgMean1), vSgScale1, vSgColor1);
    viewDependence += evalSphericalGaussian(
        directionWorld, normalize(vSgMean2), vSgScale2, vSgColor2);

    vec3 color;
    if (displayMode == DISPLAY_FULL) {
        color = diffuse + viewDependence;
    } else if (displayMode == DISPLAY_DIFFUSE) {
        color = diffuse;
    } else if (displayMode == DISPLAY_VIEW_DEPENDENT) {
        color = viewDependence;
    } else if (displayMode == DISPLAY_NORMALS) {
        color = 0.5 * (normal + 1.0);
    } else /* displayMode == DISPLAY_SHADED */ {
        color = compute_sh_shading(vec3(normal.x, normal.z, normal.y));
    }

    gl_FragColor = vec4(color, 1.0);
}
`;

/**
 * Creates a material (i.e. shaders and texture bindings) for a BakedSDF scene.
 *
 * This shader interpolates vertex attributes (normals, diffuse color, and 
 * the view dependence representation), and passes the result to the pixel shader.
 * At this stage we renormalize appropriately and shade the result.
 *
 * @return {!THREE.Material}
 */
 function createSphericalGaussianMaterial() {
  // Now pass all the 3D textures as uniforms to the shader.
  let worldspace_R_opengl = new THREE.Matrix3();
  worldspace_R_opengl['set'](
    -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 1.0, 0.0);

  const material = new THREE.ShaderMaterial({
    uniforms: {
      'worldspace_R_opengl': {'value': worldspace_R_opengl},
      'world_T_clip': {'value': new THREE.Matrix4()},
      'displayMode': {'value': gDisplayMode - 0},
    },
    vertexShader: sgVertexShaderSource,
    fragmentShader: sgFragmentShaderSource,
    vertexColors: true,
  });

  return material;
}

// Globals used for rendering.
let gControls = null; // THREE.OrbitControls
let gRenderer = null; // THREE.WebGL2Renderer
let gCamera = null; // THREE.PerspectiveCamera
let gScene = null; // THREE.Scene
let gStats = null; // Stats

function animate() {
    requestAnimationFrame(animate);
    gControls.update();

    let world_T_camera = gCamera.matrixWorld;
    let camera_T_clip = gCamera.projectionMatrixInverse;
    let world_T_clip = new THREE.Matrix4();
    world_T_clip.multiplyMatrices(world_T_camera, camera_T_clip);

    gScene.traverse(function (child) {
        if (child.isMesh) {
            if (!child.firstFrame) {
                child.frustumCulled = true;
            }
            child.firstFrame = false;
            child.material.uniforms['world_T_clip']['value'] = world_T_clip;
            child.material.uniforms['displayMode']['value'] = gDisplayMode - 0;
        }
    });
    render();
    gStats.update();
}

function render() {
    gRenderer.clear(true, true, true);
    gRenderer.render(gScene, gCamera);
}

document.addEventListener('keypress', function(e) {
    if (e.keyCode === 32 || e.key === ' ' || e.key === 'Spacebar') {
        const renderModeDiv = document.getElementById('rendermode');
        if (gDisplayMode == DisplayModeType.DISPLAY_FULL) {
            gDisplayMode = DisplayModeType.DISPLAY_DIFFUSE;
            renderModeDiv.textContent = "Diffuse only (press space to toggle)";
        } else if (gDisplayMode == DisplayModeType.DISPLAY_DIFFUSE) {
            gDisplayMode = DisplayModeType.DISPLAY_VIEW_DEPENDENT;
            renderModeDiv.textContent = "View-dependent only (press space to toggle)";
        } else if (gDisplayMode == DisplayModeType.DISPLAY_VIEW_DEPENDENT) {
            gDisplayMode = DisplayModeType.DISPLAY_NORMALS;
            renderModeDiv.textContent = "Displaying normals (press space to toggle)";
        } else if (gDisplayMode == DisplayModeType.DISPLAY_NORMALS) {
            gDisplayMode = DisplayModeType.DISPLAY_SHADED;
            renderModeDiv.textContent = "Showing shaded mesh (press space to toggle)";
        } else /* gDisplayMode == DisplayModeType.DISPLAY_SHADED */ {
            gDisplayMode = DisplayModeType.DISPLAY_FULL;
            renderModeDiv.textContent = "Full rendering (press space to toggle)";
        }
        e.preventDefault();
    }
});

function loadScene() {
    const params = new URL(window.location.href).searchParams;
    const sceneFile = params.get('scene');
    const size = params.get('s');

    const usageString =
        'To load a scene, specify the following parameters in the URL:\n' +
        '(Required) The name of the .glb scene file.\n' +
        's: (Optional) The dimensions as width,height. E.g. 640,360.';

    if (!sceneFile) {
        error('scene is a required parameter.\n\n`' + usageString);
    }

    let width = window.innerWidth - 46; // Body has a padding of 20x, we have a border of 3px.
    let height = window.innerHeight - 46;
    if (size) {
    const match = size.match(/([\d]+),([\d]+)/);
        width = parseInt(match[1], 10);
        height = parseInt(match[2], 10);
    }

    const view = create('div', 'view');
    setDims(view, width, height);
    view.textContent = '';

    const viewSpaceContainer = document.getElementById('viewspacecontainer');
    viewSpaceContainer.style.display = 'inline-block';

    const viewSpace = document.querySelector('.viewspace');
    viewSpace.textContent = '';
    viewSpace.appendChild(view);

    let canvas = document.createElement('canvas');
    view.appendChild(canvas);

    gStats = Stats();
    viewSpace.appendChild(gStats.dom);
    gStats.dom.style.position = 'absolute';

    // Set up a high performance WebGL context, making sure that anti-aliasing is
    // truned off.
    let gl = canvas.getContext('webgl2');
    gRenderer = new THREE.WebGLRenderer({
        canvas: canvas,
        context: gl,
        powerPreference: 'high-performance',
        alpha: false,
        stencil: false,
        precision: 'mediump',
        depth: false,
        antialias: false,
    });
    gRenderer.autoClear = false;
    gRenderer.sortObjects = false;
    gRenderer.setSize(view.offsetWidth, view.offsetHeight);

    gScene = new THREE.Scene();
    gCamera = new THREE.PerspectiveCamera(
        32.0, width / height, 0.25, 1000.0);
    gCamera.position.x = -0.95;
    gCamera.position.y = 0.0;
    gCamera.position.z = -0.31;

    gControls = new THREE.OrbitControls(gCamera, view);
    gControls.screenSpacePanning = true;
    gControls.enableDamping = true;
    gControls.target.x = 0.0;
    gControls.target.y = -0.3;
    gControls.target.z = 0.0;

    var loader = new THREE.GLTFLoader()
    loader.load(
        sceneFile,
        function (gltf) {
            gltf.scene.traverse(function (child) {
                if (child.isMesh) {
                    // Ensure that all meshes get uploaded on the first frame.
                    child.firstFrame = true;
                    child.frustumCulled = false; 

                    let sgMaterial = createSphericalGaussianMaterial();
                    child.material = sgMaterial;
                }
            })
            gScene.add(gltf.scene);
            hideLoading();
        },
        (xhr) => {
            console.log((xhr.loaded / xhr.total) * 100 + '% loaded');
        },
        (error) => {
            console.log(error);
        }
    );

    animate();
}

window.onload = loadScene;
