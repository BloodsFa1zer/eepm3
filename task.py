import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Задача про попит і пропозицію, варіант 15
# дані спостережень: ціна, попит, пропозиція; квота Qlim = 370
P  = np.array([2.3, 3.05, 3.7, 3.9, 4.5, 5.0, 5.8, 6.3])
Qd = np.array([700, 530, 420, 390, 330, 300, 270, 250])
Qs = np.array([100, 270, 385, 420, 490, 550, 630, 680])
Qlim = 370

def r2(y, yh):
    return 1 - ((y - yh)**2).sum() / ((y - y.mean())**2).sum()

def lsq(X, y):               # МНК: y = c0*X[0] + c1*X[1]
    return np.linalg.lstsq(np.vstack(X).T, y, rcond=None)[0]

# ---------- 1. аналітичний вигляд функцій ----------
# попит: пробуємо пряму Qd = a - b*P і гіперболу Qd = a + b/P
ad1, bd1 = lsq([np.ones_like(P), P], Qd)
ad2, bd2 = lsq([np.ones_like(P), 1 / P], Qd)
print("Попит:  пряма     Qd = %.2f %+.2f*P    R2 = %.4f" % (ad1, bd1, r2(Qd, ad1 + bd1 * P)))
print("        гіпербола Qd = %.2f + %.2f/P   R2 = %.4f" % (ad2, bd2, r2(Qd, ad2 + bd2 / P)))

# пропозиція: пробуємо пряму Qs = c + d*P і логарифм Qs = c + d*ln(P)
cs1, ds1 = lsq([np.ones_like(P), P], Qs)
cs2, ds2 = lsq([np.ones_like(P), np.log(P)], Qs)
print("Пропозиція: пряма    Qs = %.2f %+.2f*P     R2 = %.4f" % (cs1, ds1, r2(Qs, cs1 + ds1 * P)))
print("            логарифм Qs = %.2f + %.2f*lnP  R2 = %.4f" % (cs2, ds2, r2(Qs, cs2 + ds2 * np.log(P))))

# беремо форми з більшим R2
fd = lambda p: ad2 + bd2 / p            # попит
fs = lambda p: cs2 + ds2 * np.log(p)    # пропозиція
fd_inv = lambda q: bd2 / (q - ad2)      # обернена до попиту (ціна за обсягом)

# ---------- 2. ринкова рівновага і стабільність ----------
p_eq = brentq(lambda p: fd(p) - fs(p), 2, 7)
q_eq = fd(p_eq)
print("\nРівновага: P* = %.4f  Q* = %.2f" % (p_eq, q_eq))

# похідні в точці рівноваги: ринок стабільний (павутина сходиться),
# якщо попит реагує на ціну сильніше, ніж пропозиція: |Qd'| > Qs'
dd = -bd2 / p_eq**2
ds = ds2 / p_eq
print("Qd'(P*) = %.2f  Qs'(P*) = %.2f  |Qs'/Qd'| = %.3f" % (dd, ds, abs(ds / dd)))
print("|Qs'| > |Qd'| => рівновага нестабільна, павутина розходиться")

# павутиноподібна модель: Qs(P_t) продається за ціною P_{t+1} з умови Qd = Qs(P_t)
web_q, web_p = [], []
pt = p_eq - 0.2
for _ in range(6):
    qt = fs(pt)
    web_q += [qt, qt]
    web_p += [pt, fd_inv(qt)]
    pt = fd_inv(qt)
print("ціни у павутині:", np.round(web_p[1::2], 3))

# ---------- 3. квота Qlim = 370 ----------
# Qlim < Q*, тому квота дієва: обсяг = 370, ціна покупця з кривої попиту,
# ціна виробника з кривої пропозиції, різниця - рента власника квоти
p_c = fd_inv(Qlim)
p_s = brentq(lambda p: fs(p) - Qlim, 2, 7)
print("\nКвота Qlim = %d (< Q* = %.0f)" % (Qlim, q_eq))
print("ціна покупця P_d = %.4f, ціна виробника P_s = %.4f, рента = %.4f" % (p_c, p_s, p_c - p_s))

# ---------- графіки ----------
pp = np.linspace(2, 7, 300)

# якість підбору функцій
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].plot(P, Qd, 'ko', label='дані')
axs[0].plot(pp, ad1 + bd1 * pp, '--', color='gray', label='пряма')
axs[0].plot(pp, fd(pp), 'r', label='гіпербола')
axs[0].set_title('Попит')
axs[1].plot(P, Qs, 'ko', label='дані')
axs[1].plot(pp, cs1 + ds1 * pp, '--', color='gray', label='пряма')
axs[1].plot(pp, fs(pp), 'b', label='логарифм')
axs[1].set_title('Пропозиція')
for ax in axs:
    ax.set_xlabel('P')
    ax.set_ylabel('Q')
    ax.grid(alpha=.4)
    ax.legend()
plt.tight_layout()
plt.savefig('fit.png', dpi=150)
plt.close()

# ринок в осях (Q, P): рівновага і квота
plt.figure(figsize=(7.5, 5.5))
plt.plot(fd(pp), pp, 'r', label='попит D')
plt.plot(fs(pp), pp, 'b', label='пропозиція S')
plt.plot(Qd, P, 'r.', markersize=5)
plt.plot(Qs, P, 'b.', markersize=5)
plt.plot(q_eq, p_eq, 'ko')
plt.annotate('E (%.0f; %.2f)' % (q_eq, p_eq), (q_eq, p_eq), xytext=(q_eq + 30, p_eq - 0.1))
plt.axvline(Qlim, color='green', linestyle='--', label='квота Q = %d' % Qlim)
plt.plot(Qlim, p_c, 'g^', markersize=8)
plt.annotate('P_d = %.2f' % p_c, (Qlim, p_c), xytext=(Qlim + 15, p_c + 0.08))
plt.plot(Qlim, p_s, 'gv', markersize=8)
plt.annotate('P_s = %.2f' % p_s, (Qlim, p_s), xytext=(Qlim + 15, p_s - 0.2))
plt.xlim(50, 750)
plt.ylim(2, 7)
plt.xlabel('Q')
plt.ylabel('P')
plt.title('Ринкова рівновага та квота')
plt.grid(alpha=.4)
plt.legend()
plt.tight_layout()
plt.savefig('market.png', dpi=150)
plt.close()

# павутиноподібна модель
plt.figure(figsize=(7, 5))
plt.plot(fd(pp), pp, 'r', label='попит D')
plt.plot(fs(pp), pp, 'b', label='пропозиція S')
plt.plot(web_q, web_p, 'k', linewidth=0.9)
plt.plot(web_q[0], web_p[0], 'ks', markersize=6)
plt.plot(q_eq, p_eq, 'ko')
plt.xlim(250, 600)
plt.ylim(2.8, 5.2)
plt.xlabel('Q')
plt.ylabel('P')
plt.title('Павутиноподібна модель: коливання розходяться')
plt.grid(alpha=.4)
plt.legend()
plt.tight_layout()
plt.savefig('web.png', dpi=150)
plt.close()
