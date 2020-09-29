import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize


dateparse = lambda x: pd.datetime.strptime(x,'%Y%m')
portfolios = pd.read_csv('48_Industry_Portfolios.csv', skiprows=11, nrows=1128, index_col=0, parse_dates=True, date_parser=dateparse)

'Check columns names'
portfolios.columns

'5 assets portfolio'
portfolios_5 = portfolios[['Fin  ', 'Agric', 'Aero ', 'Chems', 'Hlth ']].loc['2015-07-01':'2020-06-01']

def f_frontier(data, rf, short_selling, risk_free, n):
    
    global mu_ss_rf, std_ss_rf, Z_bar, Sigma, A, B, C,delta , w_list
    
    names = data.columns
    Z_bar = np.mean(data)
    Sigma = np.cov(data.T)
    Sigma_inverse = np.linalg.inv(Sigma)
    Ones_vec = np.ones(len(Z_bar))

    A = Ones_vec.T @ Sigma_inverse @ Ones_vec
    B = Ones_vec.T @ Sigma_inverse @ Z_bar
    C = Z_bar.T @ Sigma_inverse @ Z_bar
    delta = A*C - B**2
    
    if short_selling == 'oui' :
        
        if risk_free == 'non' :
            
            mu_ss = np.linspace(-3, 3, n)
            std_ss = []
            w = np.zeros([len(mu_ss),5])

            for i in range (0, len(mu_ss)) :
                std_ss = np.append(std_ss, np.sqrt((A*mu_ss[i]**2 - 2*B*mu_ss[i] + C)/delta))

            lambda_ = (C-mu_ss[np.where(std_ss == np.min(std_ss))]*B)/delta
            gamma = (mu_ss[np.where(std_ss == np.min(std_ss))]*A-B)/delta
            w_g = lambda_*Sigma_inverse@Ones_vec + gamma*Sigma_inverse@Z_bar
            print('Le poid du portefeuille a variance minimum est :', np.round(w_g,2))

    
            # Graphique 

            plt.plot(std_ss, mu_ss)
            cmap = plt.cm.get_cmap("hsv", len(Sigma)+1)                                          #couleur
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma[i,i]), Z_bar[i], 'x', color=cmap(i))
                plt.annotate(item, (np.sqrt(Sigma[i,i]), Z_bar[i]))
            plt.axis([4, 10, -1, 2])
            plt.title('Mean variance locus, no rf')
            plt.ylabel('Return')
            plt.xlabel('σ')
            
        
        if risk_free == 'oui' :
            
            mu_ss = np.linspace(-3, 3, n)
            std_ss = []

            for i in range (0, len(mu_ss)) :
                std_ss = np.append(std_ss, np.sqrt((A*mu_ss[i]**2 - 2*B*mu_ss[i] + C)/delta))
                
            
            ## Graphique ##

            mu_ss_rf = np.linspace(-3, 3, n)
            std_ss_rf = []

            for i in range (0, len(mu_ss_rf)) :
                std_ss_rf = np.append(std_ss_rf, np.sqrt((mu_ss_rf[i]-rf)**2/(C - 2*rf*B + rf**2*A)))


            plt.axis([0, 15, -1, 2])
            plt.plot(std_ss, mu_ss)
            plt.plot(std_ss_rf, mu_ss_rf)
            cmap = plt.cm.get_cmap("hsv", len(Sigma)+1)                                          #couleur
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma[i,i]), Z_bar[i], 'x', color=cmap(i))
                plt.annotate(item, (np.sqrt(Sigma[i,i]), Z_bar[i]))
            plt.title('Mean variance locus, rf')
            plt.ylabel('Return')
            plt.xlabel('σ')
        
        
        # The tangency portfolio :


            w_t = (Sigma_inverse @ (Z_bar - rf * Ones_vec)) / (B - A*rf)
            Portfolio_mean = Z_bar.T @ w_t
            Portfolio_var = w_t.T@Sigma@w_t
            Portfolio_std = np.sqrt(Portfolio_var)
            Sharpe_ratio = np.round((Portfolio_mean - rf)/Portfolio_std,4)
            print('Les poids du portefeuilles tangent sont :', np.round(w_t,2))
            print('Le rendement espere est :', np.round(Portfolio_mean,2))
            print('La volatilite est :', np.round(Portfolio_std,2))
            print('Le ratio de Sharpe :', Sharpe_ratio)
    
    
    
    
    if short_selling == 'non' :
        
        if risk_free == 'non' :
            
            mu = np.linspace(np.min(Z_bar),np.max(Z_bar), n)
            std = []
            w_list = []

            w0 = np.ones(len(Sigma)) * 1/len(Sigma)

            bounds = [(0,1) for i in range(len(w0))]
          
            def f_variance(w):
                return 0.5*float(w.T@Sigma@w)
          
            for i in range(0, n):
                results_w = minimize(f_variance, w0, method = 'SLSQP', constraints = ({'type':'eq','fun': lambda w: float(Z_bar.T@w) - mu[i]}, {'type':'eq','fun': lambda w: np.sum(w)-1}), bounds = bounds).x
                std = np.append(std, np.sqrt(results_w.T@Sigma@results_w))
                if i >=2 :
                    if std[i] < std[i-1] :
                        w_list = np.round(results_w,4)
                        
            print('Les poids du portefeuille a variance minimum sont :', w_list)

    
            ## Graphique ##
        
        
            mu_ss = np.linspace(-3, 3, n)
            std_ss = []

            for i in range (0, len(mu_ss)) :
                std_ss = np.append(std_ss, np.sqrt((A*mu_ss[i]**2 - 2*B*mu_ss[i] + C)/delta))

            plt.plot(std_ss, mu_ss)
            cmap = plt.cm.get_cmap("hsv", len(Sigma)+1)              
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma[i,i]), Z_bar[i], 'x', color=cmap(i))
                plt.annotate(item, (np.sqrt(Sigma[i,i]), Z_bar[i]))
              
            plt.plot(std, mu)
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma[i,i]), Z_bar[i], 'x')
                plt.annotate(item, (np.sqrt(Sigma[i,i]), Z_bar[i]))  
    
            plt.axis([4, 10, -1, 2])
            plt.title('Mean variance locus, no short sell')
            plt.ylabel('Return')
            plt.xlabel('σ')
             
        if risk_free == 'oui' :
            
            mu_rf = np.linspace(np.min(Z_bar),np.max(Z_bar)*1.75, n)
            std_rf = []
            Sharpe_ratio = []
            w_list = []

            w0 = np.ones(len(Sigma)) * 1/len(Sigma)

            bounds = [(0,1000) for i in range(len(w0))]
          
            def f_variance(w):
                return 0.5*float(w.T@Sigma@w)
          
            for i in range(0, n):
                results_w = minimize(f_variance, w0, method = 'SLSQP', constraints = ({'type':'eq','fun': lambda w: float(Z_bar.T@w) + float((1-w.T@Ones_vec))*rf - mu_rf[i]}), bounds = bounds).x
                std_rf = np.append(std_rf, np.sqrt(results_w.T@Sigma@results_w))
                Sharpe_ratio = np.append(Sharpe_ratio, (mu_rf[i] - rf)/std_rf[i])
                if i >= 2 :
                    if Sharpe_ratio[i] > Sharpe_ratio[i-1] :
                        w_list = results_w
                
            Sharpe_ratio = np.max(Sharpe_ratio)
            print('Le Sharpe ratio :', np.round(Sharpe_ratio,2))
            print('Les poids du portefeuille de tangente sont :', np.round(w_list,2))
            print('Le poid dans l<actif sans risque est :', np.round(1-np.sum(w_list),2))
            
            ## Graphique ##
            
            mu = np.linspace(np.min(Z_bar),np.max(Z_bar), n)
            std = []
            w_list = []

            w0 = np.ones(len(Sigma)) * 1/len(Sigma)

            bounds = [(0,1) for i in range(len(w0))]
          
            def f_variance(w):
                return 0.5*float(w.T@Sigma@w)
          
            for i in range(0, n):
                results_w = minimize(f_variance, w0, method = 'SLSQP', constraints = ({'type':'eq','fun': lambda w: float(Z_bar.T@w) - mu[i]}, {'type':'eq','fun': lambda w: np.sum(w)-1}), bounds = bounds).x
                std = np.append(std, np.sqrt(results_w.T@Sigma@results_w))
                
            plt.plot(std, mu)
            
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma[i,i]), Z_bar[i], 'x')
                plt.annotate(item, (np.sqrt(Sigma[i,i]), Z_bar[i] ))  
              
            plt.plot(std_rf, mu_rf)
             
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma[i,i]), Z_bar[i], 'x')
                plt.annotate(item, (np.sqrt(Sigma[i,i]), Z_bar[i])) 
            

            plt.title('Mean variance locus, Rf, no short sell')
            plt.ylabel('Return')
            plt.xlabel('σ')
          
