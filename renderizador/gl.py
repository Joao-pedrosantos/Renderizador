"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: João Pedro Rodrigues dos Santos
Disciplina: Computação Gráfica
Data: 2025-08-17
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 1280
    height = 720
    near = 0.01
    far = 1000
    
    # Matrizes para pipeline 3D
    model_matrix = np.eye(4)
    view_matrix = np.eye(4)
    projection_matrix = np.eye(4)
    transform_stack = []
    
    # Dados da câmera
    camera_position = [0, 0, 0]
    camera_orientation = [0, 0, 1, 0]
    camera_fov = math.pi/4

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definir parâmetros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    # ---------- Funções de utilidade ----------
    @staticmethod
    def _rgb8(c):
        r, g, b = c if c else (1.0, 1.0, 1.0)
        if max(r, g, b) <= 1.0:
            return [int(r*255 + 0.5), int(g*255 + 0.5), int(b*255 + 0.5)]
        return [int(r), int(g), int(b)]

    @staticmethod
    def _in_bounds(u, v):
        return 0 <= u < GL.width and 0 <= v < GL.height

    @staticmethod
    def _put(u, v, col):
        if GL._in_bounds(u, v):
            gpu.GPU.draw_pixel([u, v], gpu.GPU.RGB8, col)

    @staticmethod
    def _edge(ax, ay, bx, by, px, py):
        return (px-ax)*(by-ay) - (py-ay)*(bx-ax)

    @staticmethod
    def _create_rotation_matrix(axis, angle):
        """Cria matriz de rotação usando fórmula de Rodrigues."""
        if np.linalg.norm(axis) == 0:
            return np.eye(4)
            
        axis = np.array(axis) / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        
        R = np.eye(4)
        R[:3,:3] = np.array([
            [c + axis[0]**2*(1-c), axis[0]*axis[1]*(1-c) - axis[2]*s, axis[0]*axis[2]*(1-c) + axis[1]*s],
            [axis[1]*axis[0]*(1-c) + axis[2]*s, c + axis[1]**2*(1-c), axis[1]*axis[2]*(1-c) - axis[0]*s],
            [axis[2]*axis[0]*(1-c) - axis[1]*s, axis[2]*axis[1]*(1-c) + axis[0]*s, c + axis[2]**2*(1-c)]
        ])
        return R

    @staticmethod
    def _create_perspective_matrix(fov, aspect, near, far):
        """Cria matriz de projeção perspectiva."""
        f = 1.0 / np.tan(fov / 2.0)
        P = np.zeros((4, 4))
        P[0,0] = f / aspect
        P[1,1] = f
        P[2,2] = (far + near) / (near - far)
        P[2,3] = (2 * far * near) / (near - far)
        P[3,2] = -1
        return P

    @staticmethod
    def _create_view_matrix(position, orientation):
        """Cria matriz de visualização da câmera."""
        # Posição da câmera
        eye = np.array(position)
        
        # Vetor de direção baseado na orientação
        axis = np.array(orientation[:3])
        angle = orientation[3]
        
        # Direção inicial (olhando para -Z)
        forward = np.array([0, 0, -1])
        up = np.array([0, 1, 0])
        
        # Aplica rotação da câmera
        if np.linalg.norm(axis) > 0:
            R = GL._create_rotation_matrix(axis, angle)
            forward = (R[:3,:3] @ forward)
            up = (R[:3,:3] @ up)
        
        # Calcula vetores da câmera
        target = eye + forward
        
        # Constrói matriz de visualização (lookAt)
        f = (target - eye) / np.linalg.norm(target - eye)
        s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
        u = np.cross(s, f)
        
        view = np.eye(4)
        view[:3,0] = s
        view[:3,1] = u
        view[:3,2] = -f
        view[0,3] = -np.dot(s, eye)
        view[1,3] = -np.dot(u, eye)
        view[2,3] = np.dot(f, eye)
        
        return view

    @staticmethod
    def _project_vertex(vertex):
        """Aplica pipeline completo de transformação 3D."""
        # Aplica transformações do modelo
        v_world = GL.model_matrix @ vertex
        
        # Aplica transformação da câmera
        v_view = GL.view_matrix @ v_world
        
        # Aplica projeção perspectiva
        v_clip = GL.projection_matrix @ v_view
        
        # Divisão perspectiva
        if v_clip[3] != 0:
            v_ndc = v_clip / v_clip[3]
        else:
            v_ndc = v_clip
        
        # Transformação para coordenadas de tela
        x_screen = (v_ndc[0] + 1) * GL.width / 2
        y_screen = (1 - v_ndc[1]) * GL.height / 2
        
        return x_screen, y_screen, v_ndc[2]

    @staticmethod
    def _rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
        """Rasteriza um triângulo com test de profundidade simples."""
        # Converte para inteiros
        x0, y0 = int(round(x0)), int(round(y0))
        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))
        
        # Bounding box
        minx = max(min(x0, x1, x2), 0)
        maxx = min(max(x0, x1, x2), GL.width-1)
        miny = max(min(y0, y1, y2), 0)
        maxy = min(max(y0, y1, y2), GL.height-1)
        
        # Área do triângulo
        area = GL._edge(x0, y0, x1, y1, x2, y2)
        if area <= 0:
            return
        
        # Rasterização
        inv_area = 1.0 / area
        for y in range(miny, maxy+1):
            for x in range(minx, maxx+1):
                w0 = GL._edge(x1, y1, x2, y2, x, y)
                w1 = GL._edge(x2, y2, x0, y0, x, y)
                w2 = GL._edge(x0, y0, x1, y1, x, y)

                inside = ((w0 >= 0 and w1 >= 0 and w2 >= 0) or
                        (w0 <= 0 and w1 <= 0 and w2 <= 0))
                if not inside:
                    continue

                # baricêntricas normalizadas
                alpha = w0 * inv_area
                beta  = w1 * inv_area
                gamma = w2 * inv_area

                # interpola z em NDC
                z_ndc = alpha*z0 + beta*z1 + gamma*z2
                # mapeia para [0,1] igual ao OpenGL (near=-1 -> 0, far=+1 -> 1)
                z_depth = (z_ndc + 1.0) * 0.5

                # lê z atual e compara
                z_curr = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)[0]
                if z_depth < z_curr:
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [z_depth])
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)


    # ---------- Implementações principais ----------
    
    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet 3D."""
        col = GL._rgb8(colors.get("emissiveColor"))
        
        # Processa triângulos (cada 9 valores = 3 vértices de 3 coordenadas)
        for t in range(0, len(point), 9):
            # Vértices do triângulo
            v0 = np.array([point[t], point[t+1], point[t+2], 1.0])
            v1 = np.array([point[t+3], point[t+4], point[t+5], 1.0])
            v2 = np.array([point[t+6], point[t+7], point[t+8], 1.0])
            
            # Projeta vértices
            x0, y0, z0 = GL._project_vertex(v0)
            x1, y1, z1 = GL._project_vertex(v1)
            x2, y2, z2 = GL._project_vertex(v2)
            
            # Verifica se está dentro do volume de visualização
            if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                0 <= x1 < GL.width and 0 <= y1 < GL.height and
                0 <= x2 < GL.width and 0 <= y2 < GL.height):
                GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet 3D."""
        col = GL._rgb8(colors.get("emissiveColor"))
        
        point_idx = 0
        
        # Processa cada strip
        for strip_size in stripCount:
            if strip_size < 3:  # Precisa de pelo menos 3 pontos para formar triângulos
                point_idx += strip_size * 3  # Pula os pontos desta strip
                continue
                
            # Coleta os vértices desta strip
            strip_vertices = []
            for i in range(strip_size):
                v = np.array([
                    point[point_idx + i*3],
                    point[point_idx + i*3 + 1], 
                    point[point_idx + i*3 + 2],
                    1.0
                ])
                strip_vertices.append(v)
            
            # Gera triângulos a partir da strip
            # Para cada trio consecutivo de vértices, forma um triângulo
            for i in range(strip_size - 2):
                if i % 2 == 0:  # Triângulo com orientação normal
                    v0, v1, v2 = strip_vertices[i], strip_vertices[i+1], strip_vertices[i+2]
                else:  # Triângulo com orientação invertida para manter consistência
                    v0, v1, v2 = strip_vertices[i+1], strip_vertices[i], strip_vertices[i+2]
                
                # Projeta vértices
                x0, y0, z0 = GL._project_vertex(v0)
                x1, y1, z1 = GL._project_vertex(v1)
                x2, y2, z2 = GL._project_vertex(v2)
                
                # Verifica se está dentro do volume de visualização
                if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                    0 <= x1 < GL.width and 0 <= y1 < GL.height and
                    0 <= x2 < GL.width and 0 <= y2 < GL.height):
                    GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)
            
            point_idx += strip_size * 3  # Move para próxima strip

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet 3D."""
        col = GL._rgb8(colors.get("emissiveColor"))
        
        # Converte pontos em array de vértices
        vertices = []
        for i in range(0, len(point), 3):
            v = np.array([point[i], point[i+1], point[i+2], 1.0])
            vertices.append(v)
        
        # Processa índices para formar strips
        current_strip = []
        
        for idx in index:
            if idx == -1:  # Fim de uma strip
                if len(current_strip) >= 3:
                    # Processa a strip atual
                    for i in range(len(current_strip) - 2):
                        if i % 2 == 0:  # Triângulo com orientação normal
                            i0, i1, i2 = current_strip[i], current_strip[i+1], current_strip[i+2]
                        else:  # Triângulo com orientação invertida
                            i0, i1, i2 = current_strip[i+1], current_strip[i], current_strip[i+2]
                        
                        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
                        
                        # Projeta vértices
                        x0, y0, z0 = GL._project_vertex(v0)
                        x1, y1, z1 = GL._project_vertex(v1)
                        x2, y2, z2 = GL._project_vertex(v2)
                        
                        # Verifica se está dentro do volume de visualização
                        if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                            0 <= x1 < GL.width and 0 <= y1 < GL.height and
                            0 <= x2 < GL.width and 0 <= y2 < GL.height):
                            GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)
                
                current_strip = []  # Reinicia para próxima strip
            else:
                current_strip.append(idx)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet 3D."""
        col = GL._rgb8(colors.get("emissiveColor"))
        
        # Converte coordenadas em array de vértices
        vertices = []
        for i in range(0, len(coord), 3):
            v = np.array([coord[i], coord[i+1], coord[i+2], 1.0])
            vertices.append(v)
        
        # Processa índices para formar faces
        current_face = []
        
        for idx in coordIndex:
            if idx == -1:  # Fim de uma face
                if len(current_face) >= 3:
                    # Triangula a face usando fan triangulation
                    # Conecta o primeiro vértice com todos os pares consecutivos
                    for i in range(1, len(current_face) - 1):
                        i0 = current_face[0]      # Vértice central
                        i1 = current_face[i]      # Vértice atual
                        i2 = current_face[i + 1]  # Próximo vértice
                        
                        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
                        
                        # Projeta vértices
                        x0, y0, z0 = GL._project_vertex(v0)
                        x1, y1, z1 = GL._project_vertex(v1)
                        x2, y2, z2 = GL._project_vertex(v2)
                        
                        # Verifica se está dentro do volume de visualização
                        if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                            0 <= x1 < GL.width and 0 <= y1 < GL.height and
                            0 <= x2 < GL.width and 0 <= y2 < GL.height):
                            GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)
                
                current_face = []  # Reinicia para próxima face
            else:
                current_face.append(idx)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para configurar o viewpoint (câmera)."""
        # print(f"DEBUG Viewpoint: position={position}, orientation={orientation}, fov={fieldOfView}")
        
        GL.camera_position = position
        GL.camera_orientation = orientation
        GL.camera_fov = fieldOfView
        
        # Atualiza matriz de visualização
        GL.view_matrix = GL._create_view_matrix(position, orientation)
        
        # Atualiza matriz de projeção
        aspect = GL.width / GL.height
        GL.projection_matrix = GL._create_perspective_matrix(fieldOfView, aspect, GL.near, GL.far)
        
        # print(f"DEBUG View matrix:\n{GL.view_matrix}")
        # print(f"DEBUG Projection matrix:\n{GL.projection_matrix}")

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para aplicar transformações hierárquicas."""
        # Salva estado atual na pilha
        GL.transform_stack.append(GL.model_matrix.copy())
        
        # Cria matrizes de transformação
        T = np.eye(4)  # Translação
        R = np.eye(4)  # Rotação
        S = np.eye(4)  # Escala
        
        # Matriz de escala
        if scale:
            S[0,0] = scale[0]
            S[1,1] = scale[1]
            S[2,2] = scale[2]
        
        # Matriz de rotação
        if rotation and len(rotation) == 4:
            axis = rotation[:3]
            angle = rotation[3]
            R = GL._create_rotation_matrix(axis, angle)
        
        # Matriz de translação
        if translation:
            T[0,3] = translation[0]
            T[1,3] = translation[1]
            T[2,3] = translation[2]
        
        # Combina transformações: T * R * S (ordem importante!)
        transform = T @ R @ S
        GL.model_matrix = GL.model_matrix @ transform

    @staticmethod
    def transform_out():
        """Função usada para sair de uma transformação hierárquica."""
        # Restaura estado anterior da pilha
        if GL.transform_stack:
            GL.model_matrix = GL.transform_stack.pop()
        else:
            GL.model_matrix = np.eye(4)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Box 3D."""
        col = GL._rgb8(colors.get("emissiveColor"))
        
        # Dimensões da caixa
        sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
        
        # Vértices da caixa
        vertices = [
            np.array([-sx, -sy, -sz, 1.0]),  # 0
            np.array([ sx, -sy, -sz, 1.0]),  # 1
            np.array([ sx,  sy, -sz, 1.0]),  # 2
            np.array([-sx,  sy, -sz, 1.0]),  # 3
            np.array([-sx, -sy,  sz, 1.0]),  # 4
            np.array([ sx, -sy,  sz, 1.0]),  # 5
            np.array([ sx,  sy,  sz, 1.0]),  # 6
            np.array([-sx,  sy,  sz, 1.0]),  # 7
        ]
        
        # Faces da caixa (triângulos)
        faces = [
            # Face frontal
            [0, 1, 2], [0, 2, 3],
            # Face traseira
            [5, 4, 7], [5, 7, 6],
            # Face esquerda
            [4, 0, 3], [4, 3, 7],
            # Face direita
            [1, 5, 6], [1, 6, 2],
            # Face inferior
            [4, 5, 1], [4, 1, 0],
            # Face superior
            [3, 2, 6], [3, 6, 7],
        ]
        
        # Renderiza cada face
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Projeta vértices
            x0, y0, z0 = GL._project_vertex(v0)
            x1, y1, z1 = GL._project_vertex(v1)
            x2, y2, z2 = GL._project_vertex(v2)
            
            # Verifica se está dentro do volume de visualização
            if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                0 <= x1 < GL.width and 0 <= y1 < GL.height and
                0 <= x2 < GL.width and 0 <= y2 < GL.height):
                GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Sphere 3D."""
        col = GL._rgb8(colors.get("emissiveColor"))
        
        # Parâmetros da esfera
        slices = 16  # Divisões longitudinais
        stacks = 16  # Divisões latitudinais
        
        # Gera vértices da esfera
        vertices = []
        for i in range(stacks + 1):
            phi = np.pi * i / stacks  # Ângulo latitudinal
            for j in range(slices + 1):
                theta = 2 * np.pi * j / slices  # Ângulo longitudinal
                
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.cos(phi)
                z = radius * np.sin(phi) * np.sin(theta)
                
                vertices.append(np.array([x, y, z, 1.0]))
        
        # Gera faces (triângulos)
        for i in range(stacks):
            for j in range(slices):
                # Índices dos vértices
                v0 = i * (slices + 1) + j
                v1 = v0 + 1
                v2 = (i + 1) * (slices + 1) + j
                v3 = v2 + 1
                
                # Primeiro triângulo
                if i > 0:  # Evita triângulos degenerados no polo
                    x0, y0, z0 = GL._project_vertex(vertices[v0])
                    x1, y1, z1 = GL._project_vertex(vertices[v1])
                    x2, y2, z2 = GL._project_vertex(vertices[v2])
                    
                    if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                        0 <= x1 < GL.width and 0 <= y1 < GL.height and
                        0 <= x2 < GL.width and 0 <= y2 < GL.height):
                        GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)
                
                # Segundo triângulo
                if i < stacks - 1:  # Evita triângulos degenerados no polo
                    x1, y1, z1 = GL._project_vertex(vertices[v1])
                    x2, y2, z2 = GL._project_vertex(vertices[v2])
                    x3, y3, z3 = GL._project_vertex(vertices[v3])
                    
                    if (0 <= x1 < GL.width and 0 <= y1 < GL.height and
                        0 <= x2 < GL.width and 0 <= y2 < GL.height and
                        0 <= x3 < GL.width and 0 <= y3 < GL.height):
                        GL._rasterize_triangle(x1, y1, z1, x3, y3, z3, x2, y2, z2, col)

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cone 3D."""
        col = GL._rgb8(colors.get("emissiveColor"))
        
        sides = 16  # Número de lados do cone
        
        # Vértice do topo
        top = np.array([0, height/2, 0, 1.0])
        
        # Vértices da base
        base_vertices = []
        center = np.array([0, -height/2, 0, 1.0])
        
        for i in range(sides):
            angle = 2 * np.pi * i / sides
            x = bottomRadius * np.cos(angle)
            z = bottomRadius * np.sin(angle)
            y = -height/2
            base_vertices.append(np.array([x, y, z, 1.0]))
        
        # Faces laterais
        for i in range(sides):
            next_i = (i + 1) % sides
            
            x0, y0, z0 = GL._project_vertex(top)
            x1, y1, z1 = GL._project_vertex(base_vertices[i])
            x2, y2, z2 = GL._project_vertex(base_vertices[next_i])
            
            if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                0 <= x1 < GL.width and 0 <= y1 < GL.height and
                0 <= x2 < GL.width and 0 <= y2 < GL.height):
                GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)
        
        # Base do cone
        for i in range(1, sides - 1):
            x0, y0, z0 = GL._project_vertex(center)
            x1, y1, z1 = GL._project_vertex(base_vertices[0])
            x2, y2, z2 = GL._project_vertex(base_vertices[i])
            
            if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                0 <= x1 < GL.width and 0 <= y1 < GL.height and
                0 <= x2 < GL.width and 0 <= y2 < GL.height):
                GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cylinder 3D."""
        col = GL._rgb8(colors.get("emissiveColor"))
        
        sides = 16  # Número de lados do cilindro
        
        # Vértices do topo e base
        top_vertices = []
        bottom_vertices = []
        top_center = np.array([0, height/2, 0, 1.0])
        bottom_center = np.array([0, -height/2, 0, 1.0])
        
        for i in range(sides):
            angle = 2 * np.pi * i / sides
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            top_vertices.append(np.array([x, height/2, z, 1.0]))
            bottom_vertices.append(np.array([x, -height/2, z, 1.0]))
        
        # Faces laterais
        for i in range(sides):
            next_i = (i + 1) % sides
            
            # Primeiro triângulo da face lateral
            x0, y0, z0 = GL._project_vertex(bottom_vertices[i])
            x1, y1, z1 = GL._project_vertex(top_vertices[i])
            x2, y2, z2 = GL._project_vertex(top_vertices[next_i])
            
            if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                0 <= x1 < GL.width and 0 <= y1 < GL.height and
                0 <= x2 < GL.width and 0 <= y2 < GL.height):
                GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)
            
            # Segundo triângulo da face lateral
            x0, y0, z0 = GL._project_vertex(bottom_vertices[i])
            x1, y1, z1 = GL._project_vertex(top_vertices[next_i])
            x2, y2, z2 = GL._project_vertex(bottom_vertices[next_i])
            
            if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                0 <= x1 < GL.width and 0 <= y1 < GL.height and
                0 <= x2 < GL.width and 0 <= y2 < GL.height):
                GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)
        
        # Tampa superior
        for i in range(1, sides - 1):
            x0, y0, z0 = GL._project_vertex(top_center)
            x1, y1, z1 = GL._project_vertex(top_vertices[0])
            x2, y2, z2 = GL._project_vertex(top_vertices[i + 1])
            
            if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                0 <= x1 < GL.width and 0 <= y1 < GL.height and
                0 <= x2 < GL.width and 0 <= y2 < GL.height):
                GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)
        
        # Tampa inferior
        for i in range(1, sides - 1):
            x0, y0, z0 = GL._project_vertex(bottom_center)
            x1, y1, z1 = GL._project_vertex(bottom_vertices[0])
            x2, y2, z2 = GL._project_vertex(bottom_vertices[i])
            
            if (0 <= x0 < GL.width and 0 <= y0 < GL.height and
                0 <= x1 < GL.width and 0 <= y1 < GL.height and
                0 <= x2 < GL.width and 0 <= y2 < GL.height):
                GL._rasterize_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, col)

    # ---------- Implementações 2D existentes ----------
    
    @staticmethod
    def polypoint2D(point, colors):
        col = GL._rgb8(colors.get("emissiveColor"))
        for i in range(0, len(point), 2):
            u = int(round(point[i]))
            v = int(round(point[i+1]))
            GL._put(u, v, col)

    @staticmethod
    def polyline2D(lineSegments, colors):
        col = GL._rgb8(colors.get("emissiveColor"))
        if len(lineSegments) < 4:
            return
        for i in range(0, len(lineSegments)-2, 2):
            x0, y0 = lineSegments[i], lineSegments[i+1]
            x1, y1 = lineSegments[i+2], lineSegments[i+3]
            GL._bresenham(x0, y0, x1, y1, col)

    @staticmethod
    def _bresenham(x0, y0, x1, y1, col):
        x0, y0 = int(round(x0)), int(round(y0))
        x1, y1 = int(round(x1)), int(round(y1))
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx, sy = (1, -1)[x0 > x1], (1, -1)[y0 > y1]
        err = dx - dy
        while True:
            GL._put(x0, y0, col)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2*err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    @staticmethod
    def circle2D(radius, colors):
        col = GL._rgb8(colors.get("emissiveColor"))
        cx, cy = 0.0, 0.0
        step = 1
        x_prev = cx + radius * math.sin(math.radians(0))
        y_prev = cy + radius * math.cos(math.radians(0))
        for deg in range(step, 360+step, step):
            x = cx + radius * math.sin(math.radians(deg))
            y = cy + radius * math.cos(math.radians(deg))
            GL._bresenham(x_prev, y_prev, x, y, col)
            x_prev, y_prev = x, y

    @staticmethod
    def triangleSet2D(vertices, colors):
        col = GL._rgb8(colors.get("emissiveColor"))
        for t in range(0, len(vertices), 6):
            x0, y0 = vertices[t], vertices[t+1]
            x1, y1 = vertices[t+2], vertices[t+3]
            x2, y2 = vertices[t+4], vertices[t+5]

            X0, Y0 = int(round(x0)), int(round(y0))
            X1, Y1 = int(round(x1)), int(round(y1))
            X2, Y2 = int(round(x2)), int(round(y2))

            minx = max(min(X0, X1, X2), 0)
            maxx = min(max(X0, X1, X2), GL.width-1)
            miny = max(min(Y0, Y1, Y2), 0)
            maxy = min(max(Y0, Y1, Y2), GL.height-1)

            area = GL._edge(X0, Y0, X1, Y1, X2, Y2)
            if area == 0:
                continue

            for y in range(miny, maxy+1):
                for x in range(minx, maxx+1):
                    w0 = GL._edge(X1, Y1, X2, Y2, x, y)
                    w1 = GL._edge(X2, Y2, X0, Y0, x, y)
                    w2 = GL._edge(X0, Y0, X1, Y1, x, y)
                    if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, col)

    # ---------- Funções não implementadas ou auxiliares ----------
        
    @staticmethod
    def navigationInfo(headlight):
        # print("NavigationInfo não implementado ainda")
        return 0
        
    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        print("DirectionalLight não implementado ainda")
        
    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        print("PointLight não implementado ainda")
        
    @staticmethod
    def fog(visibilityRange, color):
        print("Fog não implementado ainda")
        
    @staticmethod
    def timeSensor(cycleInterval, loop):
        epoch = time.time()
        fraction_changed = (epoch % cycleInterval) / cycleInterval
        return fraction_changed
        
    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        print("SplinePositionInterpolator não implementado ainda")
        return [0.0, 0.0, 0.0]
        
    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        print("OrientationInterpolator não implementado ainda")
        return [0, 0, 1, 0]