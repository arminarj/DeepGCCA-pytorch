import torch 

class linear_gcca():
    def __init__(self, number_views=3):
        self.U = [None for _ in range(number_views)]
        self.m = [None for _ in range(number_views)]

    def fit(self, H_list, outdim_size):

        r = 1e-4
        eps = 1e-8

        top_k = outdim_size

        AT_list =  []

        for i, H in enumerate(H_list):
            if i >=3 :
                print(i)
                assert i >=3 
            assert torch.isnan(H).sum().item() == 0 

            o_shape = H.size(0)  # N
            m = H.size(0)   # out_dim
            self.m[i] = H.mean(dim=0)
            Hbar = H - H.mean(dim=0).repeat(1, m).view(m, -1)
            assert torch.isnan(Hbar).sum().item() == 0

            A, S, B = Hbar.svd(some=True, compute_uv=True)

            A = A[:, :top_k]

            assert torch.isnan(A).sum().item() == 0

            S_thin = S[:top_k]

            S2_inv = 1. / (torch.mul( S_thin, S_thin ) + eps)

            assert torch.isnan(S2_inv).sum().item() == 0

            T2 = torch.mul( torch.mul( S_thin, S2_inv ), S_thin )

            assert torch.isnan(T2).sum().item() == 0

            T2 = torch.where(T2>eps, T2, (torch.ones(T2.shape)*eps).to(H.device).double())


            T = torch.diag(torch.sqrt(T2))

            assert torch.isnan(T).sum().item() == 0

            T_unnorm = torch.diag( S_thin + eps )

            assert torch.isnan(T_unnorm).sum().item() == 0

            AT = torch.mm(A, T)
            AT_list.append(AT)

        M_tilde = torch.cat(AT_list, dim=1)

        assert torch.isnan(M_tilde).sum().item() == 0

        Q, R = M_tilde.qr()

        assert torch.isnan(R).sum().item() == 0
        assert torch.isnan(Q).sum().item() == 0

        U, lbda, _ = R.svd(some=False, compute_uv=True)

        assert torch.isnan(U).sum().item() == 0
        assert torch.isnan(lbda).sum().item() == 0

        G = Q.mm(U[:,:top_k])
        assert torch.isnan(G).sum().item() == 0


        U = [] # Mapping from views to latent space

        # Get mapping to shared space
        views = H_list
        F = [H.shape[0] for H in H_list] # features per view
        for idx, (f, view) in enumerate(zip(F, views)):
            _, R = torch.qr(view)
            Cjj_inv = torch.inverse( (R.T.mm(R) + eps * torch.eye( view.shape[1], device=view.device)) )
            assert torch.isnan(Cjj_inv).sum().item() == 0
            pinv = Cjj_inv.mm( view.T)
                
            U.append(pinv.mm( G ))
        self.U = U 

    def _get_result(self, x, idx):
        m = x.size(0)   # out_dim
        result = x - x.mean(dim=0).repeat(1, m).view(m, -1)
        result = torch.mm(result,self.U[idx])
        return result

    def test(self, H_list):
        resualts = []
        for i, H in enumerate(H_list):
            resualts.append(self._get_result(H, i))
        return resualts