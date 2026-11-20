import math
import random
import sys
from collections import deque
import ollama
import pygame

"""
Made by Theo Kershaw (Kershaw Industries)
And maybe a little bit of Silas for the maths.
"""

WIN_W, WIN_H = 1280, 720         
PANEL_W = 350                     
SIM_W = WIN_W - PANEL_W           
FPS = 60

C_BG = (8, 9, 13)                 
C_GRID = (19, 22, 31)            
C_GRID_AXIS = (38, 44, 62)       
C_PANEL = (13, 15, 22)           
C_PANEL_EDGE = (34, 38, 54)      
C_TEXT = (205, 212, 230)         
C_DIM = (105, 114, 140)           
C_ACCENT = (0, 229, 255)          
C_ACCENT2 = (255, 61, 154)        
C_GOOD = (80, 255, 160)         
C_WARN = (255, 170, 60)        
C_POS = (255, 85, 95)           
C_NEG = (80, 145, 255)            

BODY_PIXEL_PADDING = 4

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def lerp_color(a, b, t):
    t = clamp(t, 0.0, 1.0)
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))

def fmt_num(v):
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1e5 or a < 1e-2:
        return f"{v:.3e}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.3f}".rstrip("0").rstrip(".")

def random_color():
    c = pygame.Color(0, 0, 0)
    c.hsva = (random.randint(0, 359), 65, 100, 100)
    return (c.r, c.g, c.b)

def draw_arrow(surf, color, start, end, width=2):
    pygame.draw.line(surf, color, start, end, width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 6:
        return
    ux, uy = dx / length, dy / length
    for sign in (-1, 1):                       
        ang = math.radians(150) * sign
        bx = ux * math.cos(ang) - uy * math.sin(ang)
        by = ux * math.sin(ang) + uy * math.cos(ang)
        pygame.draw.line(surf, color, end, (end[0] + bx * 7, end[1] + by * 7), width)

class BodySpec:
    def __init__(self):
        self.mass = 1.0         
        self.radius = 0.5       
        self.charge = 0.0     
        self.restitution = 0.7  
        self.friction = 0.3     
        self.omega = 0.0       
        self.pinned = False       

class Body:
    _next_id = 1

    def __init__(self, x, y, spec, px=0.0, py=0.0, color=None):
        self.id = Body._next_id
        Body._next_id += 1

        self.x, self.y = x, y            
        self.px, self.py = px, py         
        self.vx, self.vy = 0.0, 0.0      
        self.gamma = 1.0              
        self.omega = spec.omega         
        self.angle = 0.0               
        self.tau = 0.0              

        self.mass = spec.mass
        self.radius = spec.radius
        self.charge = spec.charge
        self.restitution = spec.restitution
        self.friction = spec.friction
        self.pinned = spec.pinned

        self.color = color or random_color()
        self.fx = 0.0                  
        self.fy = 0.0
        self.trail = deque(maxlen=600)   

    def inv_mass(self):
        return 0.0 if self.pinned else 1.0 / (self.mass * self.gamma)

    def inv_inertia(self):
        return 0.0 if self.pinned else 1.0 / (0.5 * self.mass * self.radius * self.radius)

    def refresh(self, c, relativistic):
        if self.pinned:
            self.vx = self.vy = 0.0
            self.gamma = 1.0
            return
        if relativistic:
            p2 = self.px * self.px + self.py * self.py
            self.gamma = math.sqrt(1.0 + p2 / (self.mass * self.mass * c * c))
        else:
            self.gamma = 1.0
        k = 1.0 / (self.mass * self.gamma)
        self.vx = self.px * k
        self.vy = self.py * k

class World:
    def __init__(self):
        self.bodies = []
        self.W = (SIM_W - 40) / 30.0      
        self.H = (WIN_H - 40) / 30.0    
        self.time = 0.0
        self.grab_body = None           
        self.grab_target = (0.0, 0.0)    
        self.defaults()

    def defaults(self):
        self.relativity = False
        self.c = 100.0               
        self.use_field = True
        self.gx, self.gy = 0.0, -9.81
        self.use_nbody = False
        self.G = 1.0     
        self.use_coulomb = False
        self.k_e = 100.0           
        self.use_mag = False
        self.B = 1.0   
        self.drag_lin = 0.0              
        self.drag_quad = 0.0         
        self.collisions = True
        self.walls = True
        self.wall_e = 0.8  
        self.time_scale = 1.0
        self.substeps = 6

    def add(self, body):
        body.refresh(self.c, self.relativity)
        self.bodies.append(body)
        return body

    def remove(self, body):
        if body in self.bodies:
            self.bodies.remove(body)
        if self.grab_body is body:
            self.grab_body = None

    def clear(self):
        self.bodies.clear()
        self.grab_body = None
        self.time = 0.0

    def body_at(self, x, y, pad):
        for b in reversed(self.bodies):
            if (b.x - x) ** 2 + (b.y - y) ** 2 <= (b.radius + pad) ** 2:
                return b
        return None
    
    def step(self, frame_dt):
        dt_total = min(frame_dt, 1.0 / 30.0) * self.time_scale
        n = max(1, int(self.substeps))
        dt = dt_total / n
        for _ in range(n):
            self.substep(dt)
        self.time += dt_total
        for b in self.bodies:
            b.trail.append((b.x, b.y))

    def substep(self, dt):
        bodies = self.bodies
        n = len(bodies)
        c, rel = self.c, self.relativity

        gb = self.grab_body
        if gb is not None and gb.pinned:
            gb.x, gb.y = self.grab_target

        for b in bodies:
            b.fx = b.fy = 0.0

        if self.use_field:
            gx, gy = self.gx, self.gy
            for b in bodies:
                b.fx += b.mass * gx
                b.fy += b.mass * gy

        if (self.use_nbody or self.use_coulomb) and n > 1:
            G = self.G if self.use_nbody else 0.0
            K = self.k_e if self.use_coulomb else 0.0
            sqrt = math.sqrt
            for i in range(n):
                a = bodies[i]
                for j in range(i + 1, n):
                    b = bodies[j]
                    dx = b.x - a.x
                    dy = b.y - a.y
                    eps = 0.5 * (a.radius + b.radius)
                    d2 = dx * dx + dy * dy + eps * eps
                    inv = 1.0 / (d2 * sqrt(d2))   
                    s = (G * a.mass * b.mass - K * a.charge * b.charge) * inv
                    if s == 0.0:
                        continue
                    fx, fy = s * dx, s * dy
                    a.fx += fx
                    a.fy += fy
                    b.fx -= fx
                    b.fy -= fy

        lin, quad = self.drag_lin, self.drag_quad
        use_mag = self.use_mag and self.B != 0.0
        ks, cd = 80.0, 2.0 * math.sqrt(80.0)
        for b in bodies:
            if b.pinned:
                continue

            if b is gb:
                tx, ty = self.grab_target
                b.fx += b.mass * (ks * (tx - b.x) - cd * b.vx)
                b.fy += b.mass * (ks * (ty - b.y) - cd * b.vy)

            b.px += b.fx * dt
            b.py += b.fy * dt

            if lin > 0.0 or quad > 0.0:
                b.refresh(c, rel)
                speed = math.hypot(b.vx, b.vy)
                decay = math.exp(-(lin + quad * speed) * dt / (b.mass * b.gamma))
                b.px *= decay
                b.py *= decay
                b.omega *= math.exp(-lin * dt / b.mass)
            if use_mag and b.charge != 0.0:
                b.refresh(c, rel)
                w = b.charge * self.B / (b.mass * b.gamma)
                th = -w * dt           
                cs, sn = math.cos(th), math.sin(th)
                b.px, b.py = b.px * cs - b.py * sn, b.px * sn + b.py * cs

            b.refresh(c, rel)
            b.x += b.vx * dt
            b.y += b.vy * dt
            b.angle += b.omega * dt
            b.tau += dt / b.gamma     

        if self.collisions and n > 1:
            self.collide_all()
        if self.walls:
            self.collide_walls()

    def collide_all(self):
        order = sorted(self.bodies, key=lambda b: b.x - b.radius)
        count = len(order)
        for i in range(count):
            a = order[i]
            right = a.x + a.radius
            for j in range(i + 1, count):
                b = order[j]
                if b.x - b.radius > right:
                    break                   
                self.resolve_pair(a, b)

    def resolve_pair(self, a, b):
        dx, dy = b.x - a.x, b.y - a.y
        rs = a.radius + b.radius
        d2 = dx * dx + dy * dy
        if d2 >= rs * rs or (a.pinned and b.pinned):
            return

        d = math.sqrt(d2)
        if d < 1e-9:                          
            nx, ny, d = 1.0, 0.0, 0.0
        else:
            nx, ny = dx / d, dy / d           
        overlap = rs - d

        ima, imb = a.inv_mass(), b.inv_mass()
        imsum = ima + imb

        corr = max(overlap - 0.001, 0.0) / imsum * 0.8
        a.x -= nx * corr * ima
        a.y -= ny * corr * ima
        b.x += nx * corr * imb
        b.y += ny * corr * imb

        rax, ray = nx * a.radius, ny * a.radius
        rbx, rby = -nx * b.radius, -ny * b.radius
        vax, vay = a.vx - a.omega * ray, a.vy + a.omega * rax
        vbx, vby = b.vx - b.omega * rby, b.vy + b.omega * rbx
        rvx, rvy = vbx - vax, vby - vay
        vn = rvx * nx + rvy * ny
        if vn > 0:
            return                           

        e = 0.5 * (a.restitution + b.restitution)
        if -vn < 0.15:
            e = 0.0                            
        jn = -(1.0 + e) * vn / imsum
        a.px -= jn * nx
        a.py -= jn * ny
        b.px += jn * nx
        b.py += jn * ny

        tx, ty = -ny, nx
        vt = rvx * tx + rvy * ty
        inva, invb = a.inv_inertia(), b.inv_inertia()
        den = imsum + a.radius ** 2 * inva + b.radius ** 2 * invb
        jt = -vt / den
        mu = math.sqrt(a.friction * b.friction)
        jt = clamp(jt, -mu * jn, mu * jn)    
        a.px -= jt * tx
        a.py -= jt * ty
        b.px += jt * tx
        b.py += jt * ty
        a.omega -= jt * a.radius * inva
        b.omega -= jt * b.radius * invb

        a.refresh(self.c, self.relativity)
        b.refresh(self.c, self.relativity)

    def collide_walls(self):
        W, H = self.W, self.H
        for b in self.bodies:
            if b.pinned:
                continue
            r = b.radius
            if b.x < r:
                b.x = r
                self._wall_hit(b, -1.0, 0.0)  
            elif b.x > W - r:
                b.x = W - r
                self._wall_hit(b, 1.0, 0.0)
            if b.y < r:
                b.y = r
                self._wall_hit(b, 0.0, -1.0)
            elif b.y > H - r:
                b.y = H - r
                self._wall_hit(b, 0.0, 1.0)

    def _wall_hit(self, b, nx, ny):
        rx, ry = nx * b.radius, ny * b.radius
        cvx, cvy = b.vx - b.omega * ry, b.vy + b.omega * rx  
        vn = cvx * nx + cvy * ny
        if vn <= 0:
            return                        
        im = b.inv_mass()
        invI = b.inv_inertia()
        e = b.restitution * self.wall_e
        if vn < 0.15:
            e = 0.0
        jn = (1.0 + e) * vn / im
        b.px -= jn * nx
        b.py -= jn * ny

        tx, ty = -ny, nx
        vt = cvx * tx + cvy * ty
        den = im + b.radius ** 2 * invI
        jt = clamp(-vt / den, -b.friction * jn, b.friction * jn)
        b.px += jt * tx
        b.py += jt * ty
        b.omega += b.radius * jt * invI     
        b.refresh(self.c, self.relativity)

    def stats(self):
        ke = 0.0
        tpx = tpy = 0.0
        vmax = 0.0
        for b in self.bodies:
            if b.pinned:
                continue
            ke += (b.px * b.px + b.py * b.py) / (b.mass * (b.gamma + 1.0))
            tpx += b.px
            tpy += b.py
            vmax = max(vmax, math.hypot(b.vx, b.vy))
        return ke, math.hypot(tpx, tpy), vmax / self.c

class Camera:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ppm = 30.0                  
        self.ox = 20.0            
        self.oy = float(WIN_H - 20)       

    def to_screen(self, x, y):
        return self.ox + x * self.ppm, self.oy - y * self.ppm

    def to_world(self, sx, sy):
        return (sx - self.ox) / self.ppm, (self.oy - sy) / self.ppm

    def zoom_at(self, sx, sy, factor):
        wx, wy = self.to_world(sx, sy)
        self.ppm = clamp(self.ppm * factor, 2.0, 400.0)
        self.ox = sx - wx * self.ppm
        self.oy = sy + wy * self.ppm


class Fonts:

    def __init__(self):
        name = "consolas,dejavusansmono,menlo,couriernew,monospace"
        self.small = pygame.font.SysFont(name, 12)
        self.norm = pygame.font.SysFont(name, 14)
        self.big = pygame.font.SysFont(name, 20, bold=True)
        self.huge = pygame.font.SysFont(name, 27, bold=True)


def blit_text(surf, font, text, pos, color=C_TEXT, right=False):
    img = font.render(text, True, color)
    r = img.get_rect()
    if right:
        r.topright = pos
    else:
        r.topleft = pos
    surf.blit(img, r)


class Toggle:
    h = 26

    def __init__(self, label, get, set_):
        self.label, self.get, self.set = label, get, set_
        self.rect = pygame.Rect(0, 0, 0, 0)

    def handle(self, ev):
        if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and self.rect.collidepoint(ev.pos)):
            self.set(not self.get())
            return True
        return False

    def draw(self, surf, fonts):
        on = self.get()
        blit_text(surf, fonts.norm, self.label, (self.rect.x, self.rect.y + 4),
                  C_TEXT if on else C_DIM)
        sw = pygame.Rect(self.rect.right - 38, self.rect.y + 4, 34, 16)
        pygame.draw.rect(surf, C_ACCENT if on else (44, 49, 68), sw, border_radius=8)
        kx = sw.right - 8 if on else sw.x + 8
        pygame.draw.circle(surf, (240, 248, 255), (kx, sw.centery), 6)


class Slider:
    h = 36

    def __init__(self, label, get, set_, lo, hi, default, log=False,
                 fmt=fmt_num, integer=False, step=None):
        self.label, self.get, self.set = label, get, set_
        self.lo, self.hi, self.default = lo, hi, default
        self.log, self.fmt, self.integer, self.step = log, fmt, integer, step
        self.drag = False
        self.rect = pygame.Rect(0, 0, 0, 0)

    def track(self):
        return pygame.Rect(self.rect.x, self.rect.y + 23, self.rect.w, 6)

    def _t(self):
        v = clamp(self.get(), self.lo, self.hi)
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def _set_from_mouse(self, mx):
        tr = self.track()
        t = clamp((mx - tr.x) / max(1, tr.w), 0.0, 1.0)
        if self.log:
            v = self.lo * (self.hi / self.lo) ** t
        else:
            v = self.lo + (self.hi - self.lo) * t
        if self.integer:
            v = int(round(v))
        elif self.step:
            v = round(v / self.step) * self.step
        self.set(v)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            if ev.button == 1:
                self.drag = True
                self._set_from_mouse(ev.pos[0])
                return True
            if ev.button == 3:
                self.set(self.default)
                return True
        elif ev.type == pygame.MOUSEMOTION and self.drag:
            self._set_from_mouse(ev.pos[0])
            return True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.drag:
            self.drag = False
            return True
        return False

    def draw(self, surf, fonts):
        blit_text(surf, fonts.norm, self.label, (self.rect.x, self.rect.y + 2), C_TEXT)
        blit_text(surf, fonts.norm, self.fmt(self.get()),
                  (self.rect.right, self.rect.y + 2), C_ACCENT, right=True)
        tr = self.track()
        pygame.draw.rect(surf, (36, 41, 58), tr, border_radius=3)
        fill = pygame.Rect(tr.x, tr.y, int(tr.w * self._t()), tr.h)
        pygame.draw.rect(surf, C_ACCENT, fill, border_radius=3)
        pygame.draw.circle(surf, (240, 248, 255), (fill.right, tr.centery), 7)


class ButtonRow:
    h = 34

    def __init__(self, items):
        self.items = items
        self.rect = pygame.Rect(0, 0, 0, 0)

    def _rects(self):
        gap = 6
        n = len(self.items)
        w = (self.rect.w - gap * (n - 1)) / n
        return [pygame.Rect(int(self.rect.x + i * (w + gap)), self.rect.y,
                            int(w), self.rect.h - 6) for i in range(n)]

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for (label, cb), r in zip(self.items, self._rects()):
                if r.collidepoint(ev.pos):
                    cb()
                    return True
        return False

    def draw(self, surf, fonts):
        mouse = pygame.mouse.get_pos()
        for (label, cb), r in zip(self.items, self._rects()):
            hot = r.collidepoint(mouse)
            pygame.draw.rect(surf, (28, 33, 48) if not hot else (36, 44, 66), r, border_radius=6)
            pygame.draw.rect(surf, C_ACCENT if hot else C_PANEL_EDGE, r, 1, border_radius=6)
            img = fonts.norm.render(label, True, C_TEXT)
            surf.blit(img, img.get_rect(center=r.center))


class Header:
    h = 28

    def __init__(self, text):
        self.text = text
        self.rect = pygame.Rect(0, 0, 0, 0)

    def handle(self, ev):
        return False

    def draw(self, surf, fonts):
        blit_text(surf, fonts.norm, self.text, (self.rect.x, self.rect.y + 8), C_ACCENT2)
        pygame.draw.line(surf, C_PANEL_EDGE, (self.rect.x, self.rect.y + 25),
                         (self.rect.right, self.rect.y + 25))


class Label:
    h = 18

    def __init__(self, fn, color=C_DIM):
        self.fn, self.color = fn, color
        self.rect = pygame.Rect(0, 0, 0, 0)

    def handle(self, ev):
        return False

    def draw(self, surf, fonts):
        blit_text(surf, fonts.small, self.fn(), (self.rect.x, self.rect.y + 2), self.color)


class Spacer:
    def __init__(self, h=8):
        self.h = h
        self.rect = pygame.Rect(0, 0, 0, 0)

    def handle(self, ev):
        return False

    def draw(self, surf, fonts):
        pass

class TextBox:
    h = 36

    def __init__(self, placeholder="", on_submit=None):
        self.text, self.placeholder, self.on_submit = "", placeholder, on_submit
        self.focused = False
        self.rect = pygame.Rect(0, 0, 0, 0)

    def submit(self):
        if self.text.strip() and self.on_submit:
            self.on_submit(self.text.strip())
        self.text = ""

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.focused = self.rect.collidepoint(ev.pos)
            return self.focused
        if self.focused and ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.submit()
            elif ev.key == pygame.K_ESCAPE:
                self.focused = False
            elif ev.unicode and ev.unicode.isprintable():
                self.text += ev.unicode
            return True
        return False

    def draw(self, surf, fonts):
        r = self.rect.inflate(0, -6)
        pygame.draw.rect(surf, (20, 24, 36), r, border_radius=6)
        pygame.draw.rect(surf, C_ACCENT if self.focused else C_PANEL_EDGE, r, 1, border_radius=6)
        shown = self.text[-32:] + ("|" if self.focused else "")
        blit_text(surf, fonts.norm, shown if (self.text or self.focused) else self.placeholder,
                  (r.x + 8, r.y + 6), C_TEXT if (self.text or self.focused) else C_DIM)


class Panel:
    TOP_Y = 112

    def __init__(self, fonts):
        self.fonts = fonts
        self.names = []
        self.tabs = {}
        self.active = 0

    def add_tab(self, name, widgets):
        self.names.append(name)
        self.tabs[name] = widgets

    def next_tab(self):
        self.active = (self.active + 1) % len(self.names)

    def tab_rects(self):
        n = len(self.names)
        gap = 4
        w = (PANEL_W - 32 - gap * (n - 1)) / n
        return [pygame.Rect(int(SIM_W + 16 + i * (w + gap)), 76, int(w), 24) for i in range(n)]

    def layout(self):
        y = self.TOP_Y
        for w in self.tabs[self.names[self.active]]:
            w.rect = pygame.Rect(SIM_W + 16, y, PANEL_W - 32, w.h)
            y += w.h

    def wants_text(self):
        return any(getattr(w, "focused", False) for w in self.tabs[self.names[self.active]])

    def blur(self):
        for w in self.tabs[self.names[self.active]]:
            if hasattr(w, "focused"):
                w.focused = False

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i, r in enumerate(self.tab_rects()):
                if r.collidepoint(ev.pos):
                    self.blur()
                    self.active = i
                    return True
        for w in self.tabs[self.names[self.active]]:
            if w.handle(ev):
                return True
        return False

    def draw(self, surf):
        f = self.fonts
        pygame.draw.rect(surf, C_PANEL, pygame.Rect(SIM_W, 0, PANEL_W, WIN_H))
        pygame.draw.line(surf, C_PANEL_EDGE, (SIM_W, 0), (SIM_W, WIN_H), 2)

        blit_text(surf, f.small, "K E R S H A W   I N D U S T R I E S", (SIM_W + 16, 12), C_ACCENT2)
        blit_text(surf, f.huge, "SINGULARITY", (SIM_W + 16, 28), C_ACCENT)
        blit_text(surf, f.small, "physics engine  |  v1.0", (SIM_W + 16, 58), C_DIM)

        for i, (name, r) in enumerate(zip(self.names, self.tab_rects())):
            on = i == self.active
            pygame.draw.rect(surf, (26, 32, 48) if on else (17, 19, 28), r, border_radius=5)
            pygame.draw.rect(surf, C_ACCENT if on else C_PANEL_EDGE, r, 1, border_radius=5)
            img = f.small.render(name, True, C_ACCENT if on else C_DIM)
            surf.blit(img, img.get_rect(center=r.center))

        self.layout()
        for w in self.tabs[self.names[self.active]]:
            w.draw(surf, f)

class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("SINGULARITY  -  Kershaw Industries Physics Engine")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H),
                                              pygame.SCALED | pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()

        self.world = World()
        self.cam = Camera()
        self.template = BodySpec()
        self.selected = None             
        self.sling = None                 

        self.paused = False
        self.step_once = False
        self.show_trails = True
        self.show_vectors = False
        self.show_grid = True
        self.show_contraction = True
        self.trail_len = 120
        self.launch_gain = 3.0            
        self.running = True

        self.scenes = [
            ("1 Sandbox", self.scene_sandbox),
            ("2 Solar", self.scene_solar),
            ("3 Charges", self.scene_charges),
            ("4 Light-Speed", self.scene_lightspeed),
            ("5 Cyclotron", self.scene_cyclotron),
        ]
        self.current_scene = self.scenes[0][1]

        self.panel = Panel(self.fonts)
        self._build_ui()
        self.scene_sandbox()

    def target(self):
        if self.selected is not None and self.selected in self.world.bodies:
            return self.selected
        self.selected = None
        return self.template

    def _build_ui(self):
        w = self.world
        app = self

        def bind(obj, name):
            return (lambda: getattr(obj, name)), (lambda v: setattr(obj, name, v))

        def tbind(name):
            return (lambda: getattr(app.target(), name)), (lambda v: setattr(app.target(), name, v))

        def set_c(v):
            w.c = v

        physics = [
            Header("RELATIVITY"),
            Toggle("Relativistic dynamics", *bind(w, "relativity")),
            Slider("Speed of light  c (m/s)", lambda: w.c, set_c, 1.0, 3.0e8, 100.0, log=True),
            ButtonRow([("c = 10", lambda: set_c(10.0)),
                       ("c = 100", lambda: set_c(100.0)),
                       ("c = REAL", lambda: set_c(299792458.0))]),
            Header("FIELDS  (N/kg = m/s^2)"),
            Toggle("Uniform gravity field", *bind(w, "use_field")),
            Slider("Field X (N/kg)", *bind(w, "gx"), -100.0, 100.0, 0.0, step=0.01),
            Slider("Field Y (N/kg)", *bind(w, "gy"), -100.0, 100.0, -9.81, step=0.01),
            Toggle("N-body gravity", *bind(w, "use_nbody")),
            Slider("G  (m^3 kg^-1 s^-2)", *bind(w, "G"), 1e-3, 1e5, 1.0, log=True),
            Toggle("Electrostatics (Coulomb)", *bind(w, "use_coulomb")),
            Slider("k  (N m^2 C^-2)", *bind(w, "k_e"), 1e-2, 1e6, 100.0, log=True),
            Toggle("Magnetic field (out of screen)", *bind(w, "use_mag")),
            Slider("B (tesla)", *bind(w, "B"), -10.0, 10.0, 1.0, step=0.01),
        ]

        world_tab = [
            Header("MEDIUM"),
            Slider("Linear drag  (kg/s)", *bind(w, "drag_lin"), 0.0, 5.0, 0.0, step=0.01),
            Slider("Quadratic drag (kg/m)", *bind(w, "drag_quad"), 0.0, 2.0, 0.0, step=0.01),
            Header("CONTACT & BOUNDARY"),
            Toggle("Body-body collisions", *bind(w, "collisions")),
            Toggle("Arena walls", *bind(w, "walls")),
            Slider("Wall bounciness", *bind(w, "wall_e"), 0.0, 1.2, 0.8, step=0.01),
            Header("TIME"),
            Slider("Time scale", *bind(w, "time_scale"), 0.01, 4.0, 1.0, log=True),
            Slider("Substeps / frame", *bind(w, "substeps"), 1, 24, 6, integer=True,
                   fmt=lambda v: str(int(v))),
            Spacer(6),
            Label(lambda: "Tip: more substeps = more accurate collisions"),
            Label(lambda: "and fast-moving bodies, but costs more CPU."),
            Label(lambda: "Right-click any slider to reset it."),
        ]

        def set_mass(v):
            t = app.target()
            if isinstance(t, Body):
                t.px *= v / t.mass
                t.py *= v / t.mass
            t.mass = v
            if isinstance(t, Body):
                t.refresh(w.c, w.relativity)

        def set_pinned(v):
            t = app.target()
            t.pinned = v
            if isinstance(t, Body) and v:
                t.px = t.py = 0.0
                t.refresh(w.c, w.relativity)

        def sel_line(i):
            b = app.selected if (app.selected in w.bodies) else None
            if b is None:
                return ["NO BODY SELECTED", "Editing the SPAWN TEMPLATE.",
                        "Click a body to edit it live.", "", "", "", ""][i]
            speed = math.hypot(b.vx, b.vy)
            ke = (b.px ** 2 + b.py ** 2) / (b.mass * (b.gamma + 1.0))
            return [
                f"BODY #{b.id}  [{'PINNED' if b.pinned else 'dynamic'}]",
                f"speed {fmt_num(speed)} m/s   v/c = {fmt_num(speed / w.c)}",
                f"gamma {b.gamma:.6f}",
                f"relativistic mass {fmt_num(b.mass * b.gamma)} kg",
                f"kinetic energy {fmt_num(ke)} J",
                f"momentum {fmt_num(math.hypot(b.px, b.py))} kg m/s",
                f"proper time {b.tau:.3f} s  (clock x{1.0 / b.gamma:.4f})",
            ][i]

        def delete_sel():
            if app.selected is not None:
                w.remove(app.selected)
                app.selected = None

        def stop_sel():
            b = app.selected
            if b is not None:
                b.px = b.py = b.omega = 0.0
                b.refresh(w.c, w.relativity)

        def clone_sel():
            b = app.selected
            if b is not None:
                spec = BodySpec()
                for name in ("mass", "radius", "charge", "restitution", "friction", "pinned"):
                    setattr(spec, name, getattr(b, name))
                nb = Body(b.x + b.radius * 2.3, b.y, spec, b.px, b.py, b.color)
                w.add(nb)
                app.selected = nb

        body_tab = [
            Header("BODY PROPERTIES"),
            Label(lambda: sel_line(0), C_ACCENT),
            Slider("Mass (kg)", lambda: app.target().mass, set_mass, 0.01, 1000.0, 1.0, log=True),
            Slider("Radius (m)", *tbind("radius"), 0.1, 3.0, 0.5, step=0.01),
            Slider("Charge (C)", *tbind("charge"), -10.0, 10.0, 0.0, step=0.1),
            Slider("Restitution (bounce)", *tbind("restitution"), 0.0, 1.0, 0.7, step=0.01),
            Slider("Friction", *tbind("friction"), 0.0, 1.0, 0.3, step=0.01),
            Slider("Spin (rad/s)", *tbind("omega"), -30.0, 30.0, 0.0, step=0.1),
            Toggle("Pinned (static)", lambda: app.target().pinned, set_pinned),
            ButtonRow([("DELETE", delete_sel), ("STOP", stop_sel), ("CLONE", clone_sel)]),
            Header("LIVE READOUT"),
            Label(lambda: sel_line(1)), Label(lambda: sel_line(2)),
            Label(lambda: sel_line(3)), Label(lambda: sel_line(4)),
            Label(lambda: sel_line(5)), Label(lambda: sel_line(6)),
        ]

        def load(fn):
            app.current_scene = fn
            fn()

        scene_widgets = [Header("SCENES  (keys 1-5)")]
        for label, fn in self.scenes:
            scene_widgets.append(ButtonRow([(label, (lambda f=fn: load(f)))]))
        scene_widgets += [
            Header("VIEW"),
            Toggle("Trails (T)", *bind(app, "show_trails")),
            Toggle("Force/velocity vectors (V)", *bind(app, "show_vectors")),
            Toggle("Grid (G)", *bind(app, "show_grid")),
            Toggle("Lorentz contraction (L)", *bind(app, "show_contraction")),
            Slider("Trail length", *bind(app, "trail_len"), 0, 600, 120, integer=True,
                   fmt=lambda v: str(int(v))),
            Slider("Slingshot gain (1/s)", *bind(app, "launch_gain"), 0.5, 10.0, 3.0, step=0.1),
            ButtonRow([("PAUSE", lambda: setattr(app, "paused", not app.paused)),
                       ("STEP", lambda: setattr(app, "step_once", True)),
                       ("CLEAR", lambda: w.clear())]),
        ]

        sage = {"answer": ""}

        def ask_sage(text):
            try:
                system_prompt = "You are Sage, you help with questions from the user involving physics and space. You will answer in around 3 sentences or shorts."
                response = ollama.chat('llama3.2:3b',
                                    messages=[
                                        {'role': 'system', 'content': system_prompt},
                                        {'role': 'user', 'content': text}
                                    ])

                sage["answer"] = response.message.content
            except Exception as e:
                sage["answer"] = e

        def sage_line(i):
            return str(sage["answer"])[i * 44:(i + 1) * 44]

        box = TextBox("Ask Sage...", on_submit=ask_sage)
        sage_tab = [
            Header("TALK TO SAGE"),
            box,
            ButtonRow([("SEND", box.submit), ("CLEAR", lambda: sage.update(answer=""))]),
            Header("REPLY"),
        ] + [Label((lambda i=i: sage_line(i)), C_TEXT) for i in range(6)]

        self.panel.add_tab("PHYSICS", physics)
        self.panel.add_tab("WORLD", world_tab)
        self.panel.add_tab("BODY", body_tab)
        self.panel.add_tab("SCENES", scene_widgets)
        self.panel.add_tab("SAGE", sage_tab)

    def _begin_scene(self):
        self.world.clear()
        self.world.defaults()
        self.cam.reset()
        self.selected = None
        self.sling = None

    def _spec(self, mass, radius, charge=0.0, e=0.7, mu=0.3, omega=0.0, pinned=False):
        s = BodySpec()
        s.mass, s.radius, s.charge = mass, radius, charge
        s.restitution, s.friction, s.omega, s.pinned = e, mu, omega, pinned
        return s

    def scene_sandbox(self):
        self._begin_scene()
        w = self.world
        for _ in range(16):
            r = random.uniform(0.3, 0.75)
            w.add(Body(random.uniform(2, w.W - 2), random.uniform(w.H * 0.4, w.H - 1),
                       self._spec(mass=4 * r * r, radius=r, e=random.uniform(0.5, 0.9),
                                  omega=random.uniform(-5, 5))))

    def scene_solar(self):
        self._begin_scene()
        w = self.world
        w.use_field = False
        w.use_nbody = True
        w.G = 1.0
        w.walls = False
        w.substeps = 10
        cx, cy = w.W / 2, w.H / 2
        M, star_r = 300.0, 0.9
        star = w.add(Body(cx, cy, self._spec(M, star_r), color=(255, 214, 90)))
        px_sum = py_sum = 0.0
        for r, m, rad in [(3.5, 1.0, 0.25), (5.5, 2.0, 0.3), (7.5, 1.5, 0.35), (10.0, 3.0, 0.45)]:
            ang = random.uniform(0, math.tau)
            eps = 0.5 * (star_r + rad)
            a = w.G * M * r / (r * r + eps * eps) ** 1.5
            v = math.sqrt(a * r)
            vx, vy = -math.sin(ang) * v, math.cos(ang) * v
            w.add(Body(cx + r * math.cos(ang), cy + r * math.sin(ang), self._spec(m, rad),
                       px=m * vx, py=m * vy))
            px_sum += m * vx
            py_sum += m * vy
        star.px, star.py = -px_sum, -py_sum
        star.refresh(w.c, w.relativity)
        self.show_trails = True
        self.trail_len = 300

    def scene_charges(self):
        self._begin_scene()
        w = self.world
        w.use_field = False
        w.use_coulomb = True
        w.k_e = 100.0
        w.drag_lin = 0.05
        w.wall_e = 1.0
        for i in range(18):
            q = 1.0 if i % 2 == 0 else -1.0
            w.add(Body(random.uniform(2, w.W - 2), random.uniform(2, w.H - 2),
                       self._spec(1.0, 0.35, charge=q, e=0.9),
                       px=random.uniform(-3, 3), py=random.uniform(-3, 3)))
        self.trail_len = 80

    def scene_lightspeed(self):
        self._begin_scene()
        w = self.world
        w.relativity = True
        w.c = 20.0
        w.gx, w.gy = 8.0, 0.0
        w.collisions = False
        w.wall_e = 0.9
        self.show_vectors = True
        self.trail_len = 200
        for i, m in enumerate([0.5, 1.0, 2.0, 5.0, 10.0]):
            w.add(Body(1.5, w.H * (i + 1) / 6.0, self._spec(m, 0.25 + 0.05 * i, e=0.9),
                       px=0.0, py=m * (i - 2) * 3.0))

    def scene_cyclotron(self):
        self._begin_scene()
        w = self.world
        w.use_field = False
        w.use_mag = True
        w.B = 1.5
        w.relativity = True
        w.c = 12.0
        w.collisions = False
        self.trail_len = 400
        for u in (2.0, 4.0, 7.0, 10.0, 14.0):
            w.add(Body(w.W / 2, w.H - 2.5, self._spec(1.0, 0.2, charge=1.0, e=1.0),
                       px=u, py=0.0))

    def spawn_from_sling(self, start, cursor_world):
        t = self.template
        sx, sy = start
        wx, wy = cursor_world
        ux = (sx - wx) * self.launch_gain
        uy = (sy - wy) * self.launch_gain
        if t.pinned:
            ux = uy = 0.0
        b = Body(sx, sy, t, px=t.mass * ux, py=t.mass * uy)
        self.world.add(b)
        self.selected = b

    def on_key(self, k):
        w = self.world
        if k == pygame.K_ESCAPE:
            self.running = False
        elif k == pygame.K_SPACE:
            self.paused = not self.paused
        elif k == pygame.K_n:
            self.step_once = True
        elif k == pygame.K_r:
            self.current_scene()
        elif k == pygame.K_c:
            w.clear()
            self.selected = None
        elif k == pygame.K_t:
            self.show_trails = not self.show_trails
        elif k == pygame.K_v:
            self.show_vectors = not self.show_vectors
        elif k == pygame.K_g:
            self.show_grid = not self.show_grid
        elif k == pygame.K_l:
            self.show_contraction = not self.show_contraction
        elif k == pygame.K_TAB:
            self.panel.next_tab()
        elif k == pygame.K_HOME:
            self.cam.reset()
        elif k in (pygame.K_DELETE, pygame.K_BACKSPACE):
            if self.selected is not None:
                w.remove(self.selected)
                self.selected = None
        elif k == pygame.K_p:
            t = self.selected
            if t is not None:
                t.pinned = not t.pinned
                t.px = t.py = 0.0
                t.refresh(w.c, w.relativity)
        elif pygame.K_1 <= k <= pygame.K_5:
            fn = self.scenes[k - pygame.K_1][1]
            self.current_scene = fn
            fn()

    def handle_event(self, ev):
        w = self.world

        if ev.type == pygame.QUIT:
            self.running = False
            return

        if ev.type in (pygame.KEYDOWN, pygame.TEXTINPUT) and self.panel.wants_text():
            self.panel.handle(ev)
            return

        if ev.type == pygame.KEYDOWN:
            self.on_key(ev.key)
            return

        if ev.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
            self.panel.handle(ev)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.pos[0] >= SIM_W:
            self.panel.handle(ev)
            return

        if ev.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if mx < SIM_W:
                self.cam.zoom_at(mx, my, 1.12 ** ev.y)

        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.pos[0] < SIM_W:
            self.panel.blur()                  
            wx, wy = self.cam.to_world(*ev.pos)
            pad = BODY_PIXEL_PADDING / self.cam.ppm
            if ev.button == 1:
                b = w.body_at(wx, wy, pad)
                if b is not None:             
                    self.selected = b
                    w.grab_body = b
                    w.grab_target = (wx, wy)
                else:                             
                    self.selected = None
                    self.sling = (wx, wy)
            elif ev.button == 3:                 
                b = w.body_at(wx, wy, pad)
                if b is not None:
                    if self.selected is b:
                        self.selected = None
                    w.remove(b)

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if w.grab_body is not None:
                w.grab_body = None               
            if self.sling is not None:
                self.spawn_from_sling(self.sling, self.cam.to_world(*ev.pos))
                self.sling = None

        elif ev.type == pygame.MOUSEMOTION and ev.buttons[1]:  
            self.cam.ox += ev.rel[0]
            self.cam.oy += ev.rel[1]

    def update(self, dt):
        w = self.world
        mx, my = pygame.mouse.get_pos()
        if w.grab_body is not None:
            if not pygame.mouse.get_pressed()[0]:
                w.grab_body = None                  
            else:
                w.grab_target = self.cam.to_world(mx, my)

        if not self.paused:
            w.step(dt)
        elif self.step_once:
            w.step(1.0 / 60.0)
        self.step_once = False

    def draw_grid(self, surf):
        cam = self.cam
        raw = 28.0 / cam.ppm
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
        x0, y1 = cam.to_world(0, 0)
        x1, y0 = cam.to_world(SIM_W, WIN_H)
        x = math.floor(x0 / step) * step
        while x <= x1:
            sx = int(cam.to_screen(x, 0)[0])
            pygame.draw.line(surf, C_GRID_AXIS if abs(x) < 1e-9 else C_GRID, (sx, 0), (sx, WIN_H))
            x += step
        y = math.floor(y0 / step) * step
        while y <= y1:
            sy = int(cam.to_screen(0, y)[1])
            pygame.draw.line(surf, C_GRID_AXIS if abs(y) < 1e-9 else C_GRID, (0, sy), (SIM_W, sy))
            y += step

    def draw_body(self, surf, b):
        w, cam = self.world, self.cam
        cx, cy = cam.to_screen(b.x, b.y)
        r = max(2.0, b.radius * cam.ppm)
        if cx < -r - 20 or cx > SIM_W + r + 20 or cy < -r - 20 or cy > WIN_H + r + 20:
            return                               

        speed = math.hypot(b.vx, b.vy)
        beta = clamp(speed / w.c, 0.0, 1.0) if w.relativity else 0.0

        col = lerp_color(b.color, (255, 255, 255), beta * 0.6)
        if beta > 0.05:
            for k in (3, 2, 1):
                halo = lerp_color(C_BG, C_ACCENT, beta * (1.0 - k / 4.0))
                pygame.draw.circle(surf, halo, (cx, cy), int(r + 3 * k), 2)

        if self.show_contraction and w.relativity and b.gamma > 1.002 and speed > 1e-9:
            sdx, sdy = b.vx / speed, -b.vy / speed   
            pts = []
            for k in range(32):
                th = math.tau * k / 32
                a_ = r * math.cos(th) / b.gamma       
                p_ = r * math.sin(th)                 
                pts.append((cx + sdx * a_ - sdy * p_, cy + sdy * a_ + sdx * p_))
            pygame.draw.polygon(surf, col, pts)
            pygame.draw.polygon(surf, lerp_color(col, (0, 0, 0), 0.45), pts, 2)
        else:
            pygame.draw.circle(surf, col, (cx, cy), int(r))
            pygame.draw.circle(surf, lerp_color(col, (0, 0, 0), 0.45), (cx, cy), int(r), 2)

        if b.charge != 0.0:
            ring = C_POS if b.charge > 0 else C_NEG
            pygame.draw.circle(surf, ring, (cx, cy), int(r), 3)
            g = max(3, int(r * 0.4))
            pygame.draw.line(surf, ring, (cx - g, cy), (cx + g, cy), 2)
            if b.charge > 0:
                pygame.draw.line(surf, ring, (cx, cy - g), (cx, cy + g), 2)

        if not b.pinned:
            pygame.draw.line(surf, (0, 0, 0), (cx, cy),
                             (cx + r * 0.85 * math.cos(b.angle), cy - r * 0.85 * math.sin(b.angle)), 2)
        else:                                     
            pygame.draw.line(surf, (0, 0, 0), (cx - r * .5, cy - r * .5), (cx + r * .5, cy + r * .5), 2)
            pygame.draw.line(surf, (0, 0, 0), (cx - r * .5, cy + r * .5), (cx + r * .5, cy - r * .5), 2)

    def draw_trail(self, surf, b):
        n = min(self.trail_len, len(b.trail))
        if n < 2:
            return
        pts = [self.cam.to_screen(x, y) for x, y in list(b.trail)[-n:]]
        stride = max(1, n // 60)                      
        last = pts[0]
        for i in range(stride, n, stride):
            pygame.draw.line(surf, lerp_color(C_BG, b.color, i / n), last, pts[i], 2)
            last = pts[i]

    def draw_vectors(self, surf, b):
        cam = self.cam
        cx, cy = cam.to_screen(b.x, b.y)

        def arrow(vx, vy, scale, color, cap):
            ex, ey = vx * cam.ppm * scale, -vy * cam.ppm * scale
            ln = math.hypot(ex, ey)
            if ln > cap:
                ex, ey = ex * cap / ln, ey * cap / ln
            if ln > 2:
                draw_arrow(surf, color, (cx, cy), (cx + ex, cy + ey))

        arrow(b.vx, b.vy, 0.25, C_GOOD, 160)          
        arrow(b.fx / b.mass, b.fy / b.mass, 0.05, C_WARN, 120)   

    def draw_sim(self, surf):
        w, cam = self.world, self.cam
        surf.fill(C_BG)
        if self.show_grid:
            self.draw_grid(surf)

        if w.walls:                                   
            x0, y0 = cam.to_screen(0, w.H)
            x1, y1 = cam.to_screen(w.W, 0)
            pygame.draw.rect(surf, (45, 140, 170), pygame.Rect(x0, y0, x1 - x0, y1 - y0), 2)

        if self.show_trails:
            for b in w.bodies:
                self.draw_trail(surf, b)
        for b in w.bodies:
            self.draw_body(surf, b)
        if self.show_vectors:
            for b in w.bodies:
                if not b.pinned:
                    self.draw_vectors(surf, b)

        if self.selected is not None and self.selected in w.bodies:
            s = self.selected
            sx, sy = cam.to_screen(s.x, s.y)
            pygame.draw.circle(surf, C_ACCENT, (sx, sy), int(s.radius * cam.ppm) + 6, 1)

        if w.grab_body is not None:
            gx, gy = cam.to_screen(w.grab_body.x, w.grab_body.y)
            pygame.draw.line(surf, C_ACCENT2, (gx, gy), pygame.mouse.get_pos(), 1)

        if self.sling is not None:
            sx, sy = cam.to_screen(*self.sling)
            mx, my = pygame.mouse.get_pos()
            t = self.template
            pygame.draw.circle(surf, C_ACCENT2, (sx, sy), max(2, int(t.radius * cam.ppm)), 1)
            pygame.draw.line(surf, C_DIM, (sx, sy), (mx, my), 1)
            draw_arrow(surf, C_ACCENT2, (sx, sy), (sx + (sx - mx), sy + (sy - my)), 2)
            wx, wy = cam.to_world(mx, my)
            u = math.hypot(self.sling[0] - wx, self.sling[1] - wy) * self.launch_gain
            blit_text(surf, self.fonts.small, f"u = {fmt_num(u)} m/s", (mx + 12, my + 8), C_ACCENT2)

    def draw_hud(self, surf):
        w, f = self.world, self.fonts
        ke, p, beta = w.stats()
        fps = self.clock.get_fps()
        lines = [
            (f"BODIES {len(w.bodies):<4} FPS {fps:5.1f}   t = {w.time:8.2f} s", C_TEXT),
            (f"KE {fmt_num(ke)} J    |p| {fmt_num(p)} kg m/s", C_DIM),
            (f"c = {fmt_num(w.c)} m/s   fastest v/c = {beta:.4f}", C_ACCENT if w.relativity else C_DIM),
        ]
        for i, (txt, col) in enumerate(lines):
            blit_text(surf, f.norm, txt, (12, 10 + i * 17), col)

        flags = []
        if w.relativity:
            flags.append("RELATIVITY")
        if w.use_field:
            flags.append(f"FIELD({fmt_num(w.gx)},{fmt_num(w.gy)})")
        if w.use_nbody:
            flags.append("N-BODY")
        if w.use_coulomb:
            flags.append("COULOMB")
        if w.use_mag:
            flags.append(f"B={fmt_num(w.B)}T")
        blit_text(surf, f.small, "  ".join(flags) or "vacuum, no forces", (12, 64), C_ACCENT2)

        if self.paused:
            blit_text(surf, f.big, "PAUSED", (SIM_W - 100, 10), C_WARN)
        blit_text(surf, f.small,
                  "LMB drag: sling/grab | RMB: delete | MMB: pan | wheel: zoom | SPACE: pause | TAB: panel",
                  (12, WIN_H - 20), C_DIM)

    def draw(self):
        self.screen.set_clip(pygame.Rect(0, 0, SIM_W, WIN_H))
        self.draw_sim(self.screen)
        self.draw_hud(self.screen)
        self.screen.set_clip(None)
        self.panel.draw(self.screen)

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            for ev in pygame.event.get():
                self.handle_event(ev)
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()

if __name__ == "__main__":
    App().run()
    sys.exit(0)
