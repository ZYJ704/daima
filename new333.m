clear; clc; close all;
rng(1);
%% =========================
% 公共参数区
%% =========================
alpha  = 0.5;      % 分数阶阶数, 0<alpha<1
lambda = 1.0;      % 记忆项系数
kappa  = 1.0;      % 扩散系数
L      = 1.0;      % 空间区间 [0,L]
T      = 1.0;      % 终止时间

M_conv = 800;      % 时间收敛表：较细空间网格
M_eff  = 400;      % 效率表：空间网格

Nlist_conv = [20, 40, 80, 160, 320, 640];
Nlist_eff  = [100, 200, 400, 800, 1600, 3200];

% DE-SOE参数
epsSOE = 1e-12;    % SOE目标精度（经验控制）
hDE    = 0.08;     % DE变量步长
nTestKernel = 300; % 核函数相对误差检测点数

% 统一绘图样式
plotcfg.LineWidth   = 1.8;
plotcfg.MarkerSize  = 7;
plotcfg.AxisFont    = 14;
plotcfg.TitleFont   = 15;
plotcfg.LegendFont  = 12;
plotcfg.DPI         = 600;

%% =========================
% 依次运行三个算例
%% =========================
for case_id = 1:3

    [case_name, outdir] = get_case_info(case_id);

    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    fprintf('========================================\n');
    fprintf('   %s\n', case_name);
    fprintf('========================================\n');

    fprintf('\n========================================\n');
    fprintf('   1) 时间收敛表：直接L1/FEM\n');
    fprintf('========================================\n');
    tbl_conv_direct = run_time_convergence_direct(alpha, lambda, kappa, L, T, M_conv, Nlist_conv, case_id);
    disp(tbl_conv_direct);

    fprintf('\n========================================\n');
    fprintf('   2) 时间收敛表：DE-SOE/FEM\n');
    fprintf('========================================\n');
    tbl_conv_soe = run_time_convergence_soe(alpha, lambda, kappa, L, T, M_conv, Nlist_conv, ...
                                            epsSOE, hDE, nTestKernel, case_id);
    disp(tbl_conv_soe);

    fprintf('\n========================================\n');
    fprintf('   3) 效率对比表：直接L1/FEM vs DE-SOE/FEM\n');
    fprintf('========================================\n');
    tbl_eff = run_efficiency_compare_DESOE(alpha, lambda, kappa, L, T, M_eff, Nlist_eff, ...
                                           epsSOE, hDE, nTestKernel, case_id);
    disp(tbl_eff);

    %% 保存表格
    writetable(tbl_conv_direct, fullfile(outdir, sprintf('time_convergence_direct_case%d.csv', case_id)));
    writetable(tbl_conv_soe,    fullfile(outdir, sprintf('time_convergence_soe_case%d.csv', case_id)));
    writetable(tbl_eff,         fullfile(outdir, sprintf('efficiency_compare_table_case%d.csv', case_id)));

    save(fullfile(outdir, sprintf('all_results_case%d.mat', case_id)), ...
         'tbl_conv_direct', 'tbl_conv_soe', 'tbl_eff', ...
         'alpha', 'lambda', 'kappa', 'L', 'T', ...
         'M_conv', 'M_eff', 'Nlist_conv', 'Nlist_eff', ...
         'epsSOE', 'hDE', 'nTestKernel', 'case_id', 'plotcfg');

    %% 作图
    plot_time_convergence_both(tbl_conv_direct, tbl_conv_soe, outdir, case_id, plotcfg);
    plot_efficiency_compare(tbl_eff, outdir, case_id, plotcfg);
    plot_speedup(tbl_eff, outdir, case_id, plotcfg);

    % 代表性N下的核逼近误差曲线
    Nrep = Nlist_eff(end);
    tau_rep = T / Nrep;
    plot_kernel_error(alpha, tau_rep, T, epsSOE, hDE, 500, outdir, case_id, plotcfg);

    fprintf('\n结果已保存到文件夹：%s\n\n', outdir);
end

fprintf('========================================\n');
fprintf('三个算例均已计算完成。\n');
fprintf('========================================\n');

%% ==========================================================
% 时间收敛表：直接L1/FEM
%% ==========================================================
function tbl = run_time_convergence_direct(alpha, lambda, kappa, L, T, M, Nlist, case_id)

    errL2 = zeros(length(Nlist),1);
    tauv  = zeros(length(Nlist),1);
    cpuv  = zeros(length(Nlist),1);

    for i = 1:length(Nlist)
        N   = Nlist(i);
        tau = T / N;
        tauv(i) = tau;

        tic;
        [x, Uend, Mmat] = solve_L1FEM_direct(alpha, lambda, kappa, L, T, M, N, case_id);
        cpuv(i) = toc;

        uex = exact_u(x, T, case_id);
        e   = Uend - uex;
        errL2(i) = sqrt(e' * Mmat * e);
    end

    order = nan(size(errL2));
    for i = 2:length(Nlist)
        order(i) = log(errL2(i-1)/errL2(i)) / log(2);
    end

    tbl = table(Nlist(:), tauv, errL2, order, cpuv, ...
        'VariableNames', {'N','tau','L2Error','Order','CPUTime'});
end

%% ==========================================================
% 时间收敛表：DE-SOE/FEM
%% ==========================================================
function tbl = run_time_convergence_soe(alpha, lambda, kappa, L, T, M, Nlist, ...
                                        epsSOE, hDE, nTestKernel, case_id)

    errL2 = zeros(length(Nlist),1);
    tauv  = zeros(length(Nlist),1);
    cpuv  = zeros(length(Nlist),1);
    Nexpv = zeros(length(Nlist),1);
    Kerr  = zeros(length(Nlist),1);

    for i = 1:length(Nlist)
        N   = Nlist(i);
        tau = T / N;
        tauv(i) = tau;

        tic;
        [x, Uend, Mmat, Nexp, kerr] = solve_FEM_L1plusDESOE(alpha, lambda, kappa, L, T, M, N, ...
                                                            epsSOE, hDE, nTestKernel, case_id);
        cpuv(i) = toc;
        Nexpv(i) = Nexp;
        Kerr(i) = kerr;

        uex = exact_u(x, T, case_id);
        e   = Uend - uex;
        errL2(i) = sqrt(e' * Mmat * e);
    end

    order = nan(size(errL2));
    for i = 2:length(Nlist)
        order(i) = log(errL2(i-1)/errL2(i)) / log(2);
    end

    tbl = table(Nlist(:), tauv, Nexpv, Kerr, errL2, order, cpuv, ...
        'VariableNames', {'N','tau','Nexp','KernelRelErr','L2Error','Order','CPUTime'});
end

%% ==========================================================
% 效率对比：直接L1/FEM vs DE-SOE/FEM
%% ==========================================================
function tbl = run_efficiency_compare_DESOE(alpha, lambda, kappa, L, T, M, Nlist, ...
                                            epsSOE, hDE, nTestKernel, case_id)

    direct_t   = zeros(length(Nlist),1);
    soe_t      = zeros(length(Nlist),1);
    direct_err = zeros(length(Nlist),1);
    soe_err    = zeros(length(Nlist),1);
    diff_err   = zeros(length(Nlist),1);
    speedup    = zeros(length(Nlist),1);
    Nexp_list  = zeros(length(Nlist),1);
    kerr_list  = zeros(length(Nlist),1);
    soe_order  = nan(length(Nlist),1);

    for i = 1:length(Nlist)
        N = Nlist(i);

        tic;
        [x1, U1, Mmat] = solve_L1FEM_direct(alpha, lambda, kappa, L, T, M, N, case_id);
        direct_t(i) = toc;

        uex1 = exact_u(x1, T, case_id);
        e1   = U1 - uex1;
        direct_err(i) = sqrt(e1' * Mmat * e1);

        tic;
        [x2, U2, Mmat2, Nexp, kerr] = solve_FEM_L1plusDESOE(alpha, lambda, kappa, L, T, M, N, ...
                                                            epsSOE, hDE, nTestKernel, case_id);
        soe_t(i) = toc;
        Nexp_list(i) = Nexp;
        kerr_list(i) = kerr;

        uex2 = exact_u(x2, T, case_id);
        e2   = U2 - uex2;
        soe_err(i) = sqrt(e2' * Mmat2 * e2);

        d12 = U1 - U2;
        diff_err(i) = sqrt(d12' * Mmat * d12);

        speedup(i) = direct_t(i) / soe_t(i);
    end

    for i = 2:length(Nlist)
        soe_order(i) = log(soe_err(i-1)/soe_err(i)) / log(2);
    end

    tbl = table(Nlist(:), Nexp_list, kerr_list, direct_t, soe_t, speedup, ...
                direct_err, soe_err, soe_order, diff_err, ...
        'VariableNames', {'N','Nexp','KernelRelErr','DirectCPU','SOECPU','Speedup', ...
                          'DirectL2Error','SOEL2Error','SOEOrder','DiffL2'});
end

%% ==========================================================
% 直接L1 + 1D线性有限元
%% ==========================================================
function [x_in, Uend, Mmat] = solve_L1FEM_direct(alpha, lambda, kappa, L, T, M, N, case_id)

    tau = T / N;

    [Mmat, Smat, x_in] = assemble_FE_1D(M, L);
    ndof = length(x_in);

    U = zeros(ndof, N+1);
    U(:,1) = exact_u(x_in, 0, case_id);

    a = zeros(N,1);
    for k = 0:N-1
        a(k+1) = (k+1)^(1-alpha) - k^(1-alpha);
    end
    b = tau^(-alpha) / gamma(2-alpha) * a;

    A = ((1/tau) + lambda * b(1)) * Mmat + kappa * Smat;
    R = chol(A, 'lower');

    for n = 1:N
        tn = n * tau;

        hist = zeros(ndof,1);
        for m = 1:n-1
            hist = hist + b(m+1) * (U(:,n-m+1) - U(:,n-m));
        end

        Fn = assemble_load_FE(M, L, @(x) source_f(x, tn, alpha, lambda, kappa, case_id));

        rhs = ((1/tau) + lambda * b(1)) * Mmat * U(:,n) ...
            - lambda * Mmat * hist ...
            + Fn;

        U(:,n+1) = R' \ (R \ rhs);
    end

    Uend = U(:,end);
end

%% ==========================================================
% L1局部项 + DE-SOE历史项
%% ==========================================================
function [x_in, Uend, Mmat, Nexp, kernelRelErr] = solve_FEM_L1plusDESOE(alpha, lambda, kappa, L, T, M, N, ...
                                                                         epsSOE, hDE, nTestKernel, case_id)

    tau = T / N;

    [Mmat, Smat, x_in] = assemble_FE_1D(M, L);
    ndof = length(x_in);

    U = zeros(ndof, N+1);
    U(:,1) = exact_u(x_in, 0, case_id);

    b0 = tau^(-alpha) / gamma(2-alpha);
    [s, w, kernelRelErr] = build_SOE_history_kernel_DE(alpha, tau, T, epsSOE, hDE, nTestKernel);
    Nexp = length(s);

    rho  = exp(-s * tau);
    beta = (1 - rho) ./ (s * tau);
    V = zeros(ndof, Nexp);

    A = ((1/tau) + lambda * b0) * Mmat + kappa * Smat;
    R = chol(A, 'lower');

    for n = 1:N
        tn = n * tau;

        Hist = zeros(ndof,1);
        for j = 1:Nexp
            Hist = Hist + (w(j) * beta(j)) * V(:,j);
        end

        Fn = assemble_load_FE(M, L, @(x) source_f(x, tn, alpha, lambda, kappa, case_id));

        rhs = ((1/tau) + lambda * b0) * Mmat * U(:,n) ...
            - lambda * Mmat * Hist ...
            + Fn;

        U(:,n+1) = R' \ (R \ rhs);

        dU = U(:,n+1) - U(:,n);
        for j = 1:Nexp
            V(:,j) = rho(j) * (V(:,j) + dU);
        end
    end

    Uend = U(:,end);
end

%% ==========================================================
% DE-SOE参数生成
%% ==========================================================
function [s, w, maxRelErr] = build_SOE_history_kernel_DE(alpha, tau, T, epsSOE, hDE, nTest)

    c = sin(pi * alpha) / pi;

    phi  = @(x) exp((pi/2) * sinh(x));
    dphi = @(x) phi(x) .* (pi/2) .* cosh(x);

    XL = 0.0;
    while true
        x = -XL;
        sx = phi(x);
        wx = c * hDE * sx^(alpha-1) * dphi(x);
        if (wx < epsSOE*1e-2) && (XL > 1.0)
            break;
        end
        XL = XL + hDE;
        if XL > 8
            break;
        end
    end

    XR = 0.0;
    while true
        x = XR;
        sx = phi(x);
        wx = c * hDE * sx^(alpha-1) * dphi(x);
        tailtau = wx * exp(-tau * sx);
        if (tailtau < epsSOE*1e-2) && (XR > 1.0)
            break;
        end
        XR = XR + hDE;
        if XR > 8
            break;
        end
    end

    xgrid = (-XL:hDE:XR)';
    s = phi(xgrid);
    w = c * hDE * s.^(alpha-1) .* dphi(xgrid);

    keep = (w .* exp(-tau * s)) > 1e-18;
    s = s(keep);
    w = w(keep);

    ttest = logspace(log10(tau), log10(T), nTest)';
    Ktrue = ttest.^(-alpha) / gamma(1-alpha);

    Kapprox = zeros(size(ttest));
    for j = 1:length(s)
        Kapprox = Kapprox + w(j) * exp(-s(j) * ttest);
    end

    relerr = abs(Kapprox - Ktrue) ./ Ktrue;
    maxRelErr = max(relerr);
end

%% ==========================================================
% 1D线性有限元矩阵组装（内部自由度）
%% ==========================================================
function [Mmat, Smat, x_in] = assemble_FE_1D(M, L)

    h = L / M;
    x_full = linspace(0, L, M+1)';
    x_in = x_full(2:end-1);

    ndof = M - 1;
    Mmat = sparse(ndof, ndof);
    Smat = sparse(ndof, ndof);

    Me = h/6 * [2 1; 1 2];
    Se = 1/h * [1 -1; -1 1];

    for e = 1:M
        nodes = [e, e+1];
        dofs = nodes - 1;

        for a = 1:2
            for b = 1:2
                ia = dofs(a);
                ib = dofs(b);
                if ia >= 1 && ia <= ndof && ib >= 1 && ib <= ndof
                    Mmat(ia, ib) = Mmat(ia, ib) + Me(a,b);
                    Smat(ia, ib) = Smat(ia, ib) + Se(a,b);
                end
            end
        end
    end
end

%% ==========================================================
% 严格有限元载荷向量组装（2点Gauss）
%% ==========================================================
function F = assemble_load_FE(M, L, fhandle)

    h = L / M;
    x = linspace(0, L, M+1)';
    ndof = M - 1;
    F = zeros(ndof,1);

    gp = [-1/sqrt(3), 1/sqrt(3)];
    gw = [1, 1];

    for e = 1:M
        xl = x(e);
        xr = x(e+1);

        Fe = zeros(2,1);

        for q = 1:2
            xi = gp(q);
            wq = gw(q);

            xq = (xl + xr)/2 + (h/2) * xi;

            phi1 = (1 - xi)/2;
            phi2 = (1 + xi)/2;

            fq = fhandle(xq);

            Fe = Fe + wq * fq * [phi1; phi2] * (h/2);
        end

        nodes = [e, e+1];
        dofs = nodes - 1;

        for a = 1:2
            ia = dofs(a);
            if ia >= 1 && ia <= ndof
                F(ia) = F(ia) + Fe(a);
            end
        end
    end
end

%% ==========================================================
% 制造解
%% ==========================================================
function u = exact_u(x, t, case_id)

    switch case_id
        case 1
            u = (t.^3) .* sin(2*pi*x);
        case 2
            u = (t.^3) .* sin(pi*x) + (t.^2) .* sin(2*pi*x);
        case 3
            u = (t.^1.5) .* x .* (1-x);
        otherwise
            error('未知算例编号 case_id.');
    end
end

%% ==========================================================
% 源项
%% ==========================================================
function f = source_f(x, t, alpha, lambda, kappa, case_id)

    if t == 0
        f = zeros(size(x));
        return;
    end

    switch case_id
        case 1
            term1 = 3 * t^2 .* sin(2*pi*x);
            term2 = lambda * 6 / gamma(4 - alpha) * t^(3 - alpha) .* sin(2*pi*x);
            term3 = 4 * kappa * pi^2 * t^3 .* sin(2*pi*x);
            f = term1 + term2 + term3;

        case 2
            term1 = 3 * t^2 .* sin(pi*x) + 2 * t .* sin(2*pi*x);
            term2 = lambda * ( ...
                    6 / gamma(4 - alpha) * t^(3 - alpha) .* sin(pi*x) ...
                  + 2 / gamma(3 - alpha) * t^(2 - alpha) .* sin(2*pi*x) );
            term3 = kappa * ( ...
                    pi^2 * t^3 .* sin(pi*x) ...
                  + 4 * pi^2 * t^2 .* sin(2*pi*x) );
            f = term1 + term2 + term3;

        case 3
            term1 = 1.5 * t^0.5 .* x .* (1-x);
            term2 = lambda * gamma(2.5) / gamma(2.5 - alpha) * t^(1.5 - alpha) .* x .* (1-x);
            term3 = 2 * kappa * t^1.5 * ones(size(x));
            f = term1 + term2 + term3;

        otherwise
            error('未知算例编号 case_id.');
    end
end

%% ==========================================================
% 算例信息
%% ==========================================================
function [case_name, outdir] = get_case_info(case_id)

    switch case_id
        case 1
            case_name = '算例1: u(x,t)=t^3 sin(2*pi*x)';
            outdir    = 'results_case1_t3sin2pix_DE_SOE_中文高清增强版';

        case 2
            case_name = '算例2: u(x,t)=t^3 sin(pi*x)+t^2 sin(2*pi*x)';
            outdir    = 'results_case2_t3sinpix_plus_t2sin2pix_DE_SOE_中文高清增强版';

        case 3
            case_name = '算例3: u(x,t)=t^{1.5} x (1-x)';
            outdir    = 'results_case3_t15_x1mx_DE_SOE_中文高清增强版';

        otherwise
            error('未知算例编号 case_id.');
    end
end

%% ==========================================================
% 绘图：时间收敛（双方法）
%% ==========================================================
function plot_time_convergence_both(tbl_direct, tbl_soe, outdir, case_id, cfg)

    f = newfig();
    loglog(tbl_direct.tau, tbl_direct.L2Error, 'o-', ...
        'LineWidth', cfg.LineWidth, 'MarkerSize', cfg.MarkerSize); hold on;
    loglog(tbl_soe.tau, tbl_soe.L2Error, 's-', ...
        'LineWidth', cfg.LineWidth, 'MarkerSize', cfg.MarkerSize);

    ref = tbl_direct.L2Error(1) * (tbl_direct.tau / tbl_direct.tau(1)).^1;
    loglog(tbl_direct.tau, ref, 'k--', 'LineWidth', 1.2);

    grid on;
    xlabel('时间步长 \tau', 'FontSize', cfg.AxisFont);
    ylabel('终时刻 L2 误差', 'FontSize', cfg.AxisFont);
    legend({'直接L1法','DE-SOE法','一阶参考线'}, 'Location', 'best', 'FontSize', cfg.LegendFont);
    title(sprintf('算例%d：时间收敛曲线', case_id), 'FontSize', cfg.TitleFont);
    set(gca, 'FontSize', cfg.AxisFont, 'LineWidth', 1.0);
    exportgraphics(f, fullfile(outdir, sprintf('时间收敛_双方法_算例%d.png', case_id)), 'Resolution', cfg.DPI);
    close(f);
end

%% ==========================================================
% 绘图：CPU时间对比
%% ==========================================================
function plot_efficiency_compare(tbl_eff, outdir, case_id, cfg)

    f = newfig();
    loglog(tbl_eff.N, tbl_eff.DirectCPU, 'o-', ...
        'LineWidth', cfg.LineWidth, 'MarkerSize', cfg.MarkerSize); hold on;
    loglog(tbl_eff.N, tbl_eff.SOECPU, 's-', ...
        'LineWidth', cfg.LineWidth, 'MarkerSize', cfg.MarkerSize);
    grid on;
    xlabel('时间步数 N', 'FontSize', cfg.AxisFont);
    ylabel('CPU时间 / s', 'FontSize', cfg.AxisFont);
    legend({'直接L1法','DE-SOE法'}, 'Location', 'northwest', 'FontSize', cfg.LegendFont);
    title(sprintf('算例%d：CPU时间对比', case_id), 'FontSize', cfg.TitleFont);
    set(gca, 'FontSize', cfg.AxisFont, 'LineWidth', 1.0);
    exportgraphics(f, fullfile(outdir, sprintf('CPU时间对比_算例%d.png', case_id)), 'Resolution', cfg.DPI);
    close(f);
end

%% ==========================================================
% 绘图：加速比
%% ==========================================================
function plot_speedup(tbl_eff, outdir, case_id, cfg)

    f = newfig();
    semilogx(tbl_eff.N, tbl_eff.Speedup, 'o-', ...
        'LineWidth', cfg.LineWidth, 'MarkerSize', cfg.MarkerSize);
    grid on;
    xlabel('时间步数 N', 'FontSize', cfg.AxisFont);
    ylabel('加速比（直接法CPU / SOE法CPU）', 'FontSize', cfg.AxisFont);
    title(sprintf('算例%d：DE-SOE/FEM加速比', case_id), 'FontSize', cfg.TitleFont);
    set(gca, 'FontSize', cfg.AxisFont, 'LineWidth', 1.0);
    exportgraphics(f, fullfile(outdir, sprintf('加速比_算例%d.png', case_id)), 'Resolution', cfg.DPI);
    close(f);
end

%% ==========================================================
% 绘图：核函数相对误差
%% ==========================================================
function plot_kernel_error(alpha, tau, T, epsSOE, hDE, nTest, outdir, case_id, cfg)

    [s, w, ~] = build_SOE_history_kernel_DE(alpha, tau, T, epsSOE, hDE, nTest);

    ttest = logspace(log10(tau), log10(T), nTest)';
    Ktrue = ttest.^(-alpha) / gamma(1-alpha);

    Kapprox = zeros(size(ttest));
    for j = 1:length(s)
        Kapprox = Kapprox + w(j) * exp(-s(j) * ttest);
    end

    relerr = abs(Kapprox - Ktrue) ./ Ktrue;

    f = newfig();
    loglog(ttest, relerr, '-', 'LineWidth', cfg.LineWidth);
    grid on;
    xlabel('时间 t', 'FontSize', cfg.AxisFont);
    ylabel('相对误差', 'FontSize', cfg.AxisFont);
    title(sprintf('算例%d：DE-SOE核函数逼近相对误差', case_id), 'FontSize', cfg.TitleFont);
    set(gca, 'FontSize', cfg.AxisFont, 'LineWidth', 1.0);
    exportgraphics(f, fullfile(outdir, sprintf('核函数相对误差_算例%d.png', case_id)), 'Resolution', cfg.DPI);
    close(f);
end

%% ==========================================================
% 统一新建图窗
%% ==========================================================
function f = newfig()
    f = figure('Color','w', ...
               'Position',[100 100 900 650], ...
               'Visible','off');
end