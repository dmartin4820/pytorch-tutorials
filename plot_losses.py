import pickle
import matplotlib.pyplot as plt


g_losses_fn = "data/output/g_losses.pickle"
d_losses_fn = "data/output/d_losses.pickle"

with open(g_losses_fn, 'rb') as gfn:
  G_losses = pickle.load(gfn)

with open(d_losses_fn, 'rb') as dfn:
  D_losses = pickle.load(dfn)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
