import numpy as np

# ta chon ham f(x,y)= x^2 + 2y^2 - 4x + 4y
def f(x):
    return x[0]**2 + 2*x[1]**2 - 4*x[0] + 4*x[1]

# dao ham bac 1
def grad(x):
    return np.array([2*x[0] - 4, 4*x[1] + 4])

# dao ham bac 2 (hay hessian)
# vi la ham bac 2 nen hessian la hang so ko phu thuoc vao x.
def hess(x):
    return np.array([[2, 0], 
                     [0, 4]])

def newton_raphson(x_bd, sai_so=1e-6, vong_lap=100):
    x = np.array(x_bd, dtype=float)
    
    for i in range(vong_lap):
        g = grad(x)
        H = hess(x)
        
        if np.linalg.norm(g) < sai_so:
            break
            
        H_inv = np.linalg.inv(H)
        x_moi = x - np.dot(H_inv, g)
        
        if np.linalg.norm(x_moi - x) < sai_so:
            x = x_moi
            break
            
        x = x_moi
        
        # in ra gia tri f(x) de xem no co giam hay ko.
        print(f"Buoc {i+1}: x = {x}, f(x) = {f(x)}")
        
    return x

if __name__ == "__main__":
    x_start = [0.0, 0.0]
    kq = newton_raphson(x_start)
    
    print("\n=> Diem cuc tri tim duoc:", kq)
    # test thu xem la cuc tieu hay cuc dai bang tri rieng
    eig_vals = np.linalg.eigvals(hess(kq))
    if np.all(eig_vals > 0):
        print("Day la diem cuc tieu do cac tri rieng > 0")
    elif np.all(eig_vals < 0):
        print("Day la diem cuc dai")
    else:
        print("Diem yen ngua")