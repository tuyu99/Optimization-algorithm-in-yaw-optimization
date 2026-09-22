
import warnings
from time import perf_counter as timerpc

import numpy as np

from floris.logging_manager import LoggingManager

# from .yaw_optimizer_scipy import YawOptimizationScipy
from .yaw_optimization_base import YawOptimization


class YawOptimizationSR(YawOptimization, LoggingManager):
    """
    Switch between original SR and a pass-based GSR variant.

    GSR differs intentionally from Tu et al., Algorithm 1 / Table A.6:
    every turbine's interval is recomputed on EVERY pass, regardless of improvement.
    Define initial_half_width = (upper-lower)/2 using the ORIGINAL bounds.
    Pass p (one-based): full bounds for p=1; otherwise radius
    initial_half_width / gsr_alpha**(p-1) about the current best angle.
    Intersect with original bounds, then enumerate ceil(lower)..floor(upper).
    For original [-30, 30] and incumbent 0, integer candidate intervals are:
      alpha=2: [-30, 30], [-15, 15], [-7, 7].
      alpha=3: [-30, 30], [-10, 10], [-3, 3].
    Nonzero incumbents shift the center; fixed bounds cannot shrink further.
    gsr_alpha=1 explicitly searches the full bounds on every pass.
    Enumerate integer degrees inside the interval, keeping the incumbent as well.
    If an interval contains no integer, evaluate its endpoints instead (a practical
    extension for fractional/fixed constraints). No global-optimality guarantee.

    Import this module explicitly to use the switch. It works with FlorisModel,
    UncertainFlorisModel and ParFlorisModel through the inherited power interface.
    The old ParallelFlorisModel.optimize_yaw_angles wrapper does not forward the
    new options; use ParFlorisModel with this class instead.
    Like the original SR, x0 and downstream exclusion do not drive the search;
    fixed bounds and disabled-turbine bounds are respected. Downstream turbines
    are NOT automatically skipped, avoiding a new nominal-direction heuristic
    in uncertain wind conditions.

    See :cite:`fleming_sr_2022` for full details on the SR method.
    """
    def __init__(
        self,
        fmodel,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=None,  # SR only; None means [5, 4]
        turbine_weights=None,
        exclude_downstream_turbines=True,
        verify_convergence=False,
        *,
        mode="sr",
        gsr_n_passes=3,
        gsr_alpha=2.0,
    ):
        """
        Instantiate YawOptimizationSR object with a FlorisModel object
        and assign parameter values.

        Args:
            fmodel: An instantiated FlorisModel object.
            minimum_yaw_angle: Minimum yaw angle for all turbines [degrees]. Default is 0.0.
            maximum_yaw_angle: Maximum yaw angle for all turbines [degrees]. Default is 25.0.
            yaw_angles_baseline: Yaw angles to use as a baseline for comparison to optimized
               yaw angles [degrees]. If None, defaults to 0.0 for all turbines.
            x0: Not used in this optimizer. Included for compatibility with base class. Defaults to
                None.
            Ny_passes: List of integers defining the number of yaw angles to evaluate
                per turbine in each pass of the SR algorithm. The length of the list
                defines the number of passes. The first entry can be even or odd,
                but all further entries must be even. Default is [5, 4].
            turbine_weights: Weights for each turbine when calculating the
                weighted power output during optimization. If None, all turbines
                are weighted equally. Default is None.
            exclude_downstream_turbines: Not used in this optimizer. Included for compatibility with
                base class. Default is True.
            verify_convergence: If True, the optimizer will perform additional checks to verify
                that the optimal yaw angles have been found. See
                YawOptimization._verify_solutions_for_convergence() for more details.
            mode: "sr" (default) preserves the original SR search; "gsr" selects
                unconditional pass-based refinement. This variant does not reproduce
                Algorithm 1's successful-improvement counter.
            gsr_n_passes: Positive integer, number of complete GSR sweeps. Default 3.
            gsr_alpha: Finite number >= 1, default 2.0. At 1, search the entire
                feasible interval every pass; above 1, shrink by pass number.
                GSR ignores Ny_passes and warns if it is explicitly supplied.
        """

        # Validate before expensive model initialization. Keep old positional order.
        if mode not in ("sr", "gsr"):
            raise ValueError("mode must be 'sr' or 'gsr'.")
        self.mode = mode
        if (isinstance(gsr_n_passes, (bool, np.bool_)) or
                not isinstance(gsr_n_passes, (int, np.integer)) or gsr_n_passes < 1):
            raise ValueError("gsr_n_passes must be a positive integer.")
        if (isinstance(gsr_alpha, (bool, np.bool_)) or
                not isinstance(gsr_alpha, (int, float, np.integer, np.floating)) or
                not np.isfinite(gsr_alpha) or gsr_alpha < 1):
            raise ValueError("gsr_alpha must be finite and >= 1.")
        self.gsr_n_passes = int(gsr_n_passes)
        self.gsr_alpha = float(gsr_alpha)
        if mode == "gsr" and Ny_passes is not None:
            warnings.warn("Ny_passes is SR-only and is ignored in GSR mode.", UserWarning)
        Ny_passes = [5, 4] if Ny_passes is None or mode == "gsr" else list(Ny_passes)
        if not Ny_passes:
            raise ValueError("Ny_passes must not be empty.")

        # Warn if non-default values are provided for unused inputs
        if x0 is not None:
            warnings.warn(
                "The 'x0' argument is not used in the Serial Refine optimization method "
                "and will be ignored.",
                UserWarning
            )

        # Initialize base class
        super().__init__(
            fmodel=fmodel,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            turbine_weights=turbine_weights,
            calc_baseline_power=True,
            exclude_downstream_turbines=exclude_downstream_turbines,
            verify_convergence=verify_convergence,
        )

        if mode == "gsr":
            lb, ub = self._minimum_yaw_angle_subset, self._maximum_yaw_angle_subset
            if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))) or np.any(lb > ub):
                raise ValueError("GSR requires finite, ordered yaw bounds.")
            # Baseline is a reference, which may lie outside feasible bounds.
            feasible = np.clip(self._yaw_angles_opt_subset, lb, ub)
            if not np.array_equal(feasible, self._yaw_angles_opt_subset):
                self._yaw_angles_opt_subset = feasible
                self._farm_power_opt_subset = self._calculate_farm_power(
                    yaw_angles=feasible,
                    power_setpoints=self.fmodel_subset.core.farm.power_setpoints,
                )
            if not np.all(np.isfinite(self._farm_power_opt_subset)):
                raise ValueError("GSR requires finite initial farm powers.")

        # Start a timer for FLORIS computations
        self.time_spent_in_floris = 0

        # Confirm that Ny_passes are integers and odd/even
        for Nii, Ny in enumerate(Ny_passes):
            if not isinstance(Ny, int):
                raise ValueError("Ny_passes must contain exclusively integers")
            if Ny < 2:
                raise ValueError("Each entry in Ny_passes must have a value of at least 2.")
            if (Nii > 0) & ((Ny + 1) % 2 == 0):
                raise ValueError(
                    "The second and further entries of Ny_passes must be even numbers. "
                    "This is to ensure the same yaw angles are not evaluated twice between passes."
                )

        # Save optimization choices to self
        self.Ny_passes = Ny_passes

        # For each wind direction, determine the order of turbines
        self._get_turbine_orders()

    def _get_turbine_orders(self):
        layout_x = self.fmodel.layout_x
        layout_y = self.fmodel.layout_y
        turbines_ordered_array = []
        for wd in self.fmodel_subset.core.flow_field.wind_directions:
            layout_x_rot = (
                np.cos((wd - 270.0) * np.pi / 180.0) * layout_x
                - np.sin((wd - 270.0) * np.pi / 180.0) * layout_y
            )
            if self.mode == "gsr":
                angle = np.deg2rad(wd - 270.0)
                y_rot = np.sin(angle) * layout_x + np.cos(angle) * layout_y
                # Round numerical rotation noise when defining equal streamwise positions.
                turbines_ordered = np.lexsort((np.arange(self.nturbs), y_rot,
                                               np.round(layout_x_rot, 8)))
            else:
                turbines_ordered = np.argsort(layout_x_rot)
            turbines_ordered_array.append(turbines_ordered)
        self.turbines_ordered_array_subset = np.vstack(turbines_ordered_array)


    def _calc_powers_with_memory(self, yaw_angles_subset, use_memory=True):
        # Define current optimal solutions and floris wind directions locally
        yaw_angles_opt_subset = self._yaw_angles_opt_subset
        farm_power_opt_subset = self._farm_power_opt_subset
        wd_array_subset = self.fmodel_subset.core.flow_field.wind_directions
        ws_array_subset = self.fmodel_subset.core.flow_field.wind_speeds
        ti_array_subset = self.fmodel_subset.core.flow_field.turbulence_intensities
        power_setpoints_subset = self.fmodel_subset.core.farm.power_setpoints
        turbine_weights_subset = self._turbine_weights_subset

        # Reformat yaw_angles_subset, if necessary
        Ny = 1
        eval_multiple_passes = (len(np.shape(yaw_angles_subset)) == 3)
        if eval_multiple_passes:
            # Four-dimensional; format everything into three-dimensional
            Ny = yaw_angles_subset.shape[0]  # Number of passes
            yaw_angles_subset = np.vstack(
                [yaw_angles_subset[iii, :, :] for iii in range(Ny)]
            )
            yaw_angles_opt_subset = np.tile(yaw_angles_opt_subset, (Ny, 1))
            farm_power_opt_subset = np.tile(farm_power_opt_subset, (Ny))
            wd_array_subset = np.tile(wd_array_subset, Ny)
            ws_array_subset = np.tile(ws_array_subset, Ny)
            ti_array_subset = np.tile(ti_array_subset, Ny)
            power_setpoints_subset = np.tile(power_setpoints_subset, (Ny, 1))
            turbine_weights_subset = np.tile(turbine_weights_subset, (Ny, 1))

        # Initialize empty matrix for floris farm power outputs
        farm_powers = np.zeros((yaw_angles_subset.shape[0]))

        # Find indices of yaw angles that we previously already evaluated, and
        # prevent redoing the same calculations
        if use_memory:
            if self.mode == "gsr":
                idx = (yaw_angles_opt_subset == yaw_angles_subset).all(axis=1)
            else:
                idx = (np.abs(yaw_angles_opt_subset - yaw_angles_subset) < 0.01).all(axis=1)
            farm_powers[idx] = farm_power_opt_subset[idx]
            if self.print_progress:
                self.logger.info(
                    "Skipping {:d}/{:d} calculations: already in memory.".format(
                        np.sum(idx), len(idx))
                )
        else:
            idx = np.zeros(yaw_angles_subset.shape[0], dtype=bool)

        if not np.all(idx):
            # Now calculate farm powers for conditions we haven't yet evaluated previously
            start_time = timerpc()
            if (hasattr(self.fmodel.core.flow_field, 'heterogeneous_inflow_config') and
                self.fmodel.core.flow_field.heterogeneous_inflow_config is not None):
                het_sm_orig = np.array(
                    self.fmodel.core.flow_field.heterogeneous_inflow_config['speed_multipliers']
                )
                het_sm = np.tile(het_sm_orig, (Ny, 1))[~idx, :]
            else:
                het_sm = None
            farm_powers[~idx] = self._calculate_farm_power(
                wd_array=wd_array_subset[~idx],
                ws_array=ws_array_subset[~idx],
                ti_array=ti_array_subset[~idx],
                turbine_weights=turbine_weights_subset[~idx, :],
                yaw_angles=yaw_angles_subset[~idx, :],
                heterogeneous_speed_multipliers=het_sm,
                power_setpoints=power_setpoints_subset[~idx, :],
            )
            self.time_spent_in_floris += (timerpc() - start_time)

        # Finally format solutions back to original format, if necessary
        if eval_multiple_passes:
            farm_powers = np.reshape(
                farm_powers,
                (
                    Ny,
                    self.fmodel_subset.core.flow_field.n_findex
                )
            )

        return farm_powers

    def _generate_evaluation_grid(self, pass_depth, turbine_depth):
        """
        Calculate the yaw angles for every iteration in the SR algorithm, for turbine,
        for every wind direction, for every wind speed, for every TI. Basically, this
        should yield a grid of yaw angle sets to evaluate the wind farm AEP with 'Ny'
        times. Then, for each ambient condition set,
        """

        # Initialize yaw angles to evaluate, 'Ny' times the wind rose
        Ny = self.Ny_passes[pass_depth]
        evaluation_grid = np.tile(self._yaw_angles_opt_subset, (Ny, 1, 1))

        # Get a list of the turbines in order of x and sort front to back
        for iw in range(self._n_findex_subset):
            turbid = self.turbines_ordered_array_subset[iw, turbine_depth]  # Turbine to manipulate

            # Grab yaw bounds from self
            yaw_lb = self._yaw_lbs[iw, turbid]
            yaw_ub = self._yaw_ubs[iw, turbid]

            # Saturate to allowable yaw limits
            yaw_lb = np.clip(
                yaw_lb,
                self.minimum_yaw_angle[iw, turbid],
                self.maximum_yaw_angle[iw, turbid]
            )
            yaw_ub = np.clip(
                yaw_ub,
                self.minimum_yaw_angle[iw, turbid],
                self.maximum_yaw_angle[iw, turbid]
            )

            if pass_depth == 0:
                # Evaluate all possible coordinates
                yaw_angles_subset = np.linspace(yaw_lb, yaw_ub, Ny)
            else:
                # Remove middle point: was evaluated in previous iteration
                c = int(Ny / 2)  # Central point (to remove)
                ids = [*list(range(0, c)), *list(range(c + 1, Ny + 1))]
                yaw_angles_subset = np.linspace(yaw_lb, yaw_ub, Ny + 1)[ids]

            evaluation_grid[:, iw, turbid] = yaw_angles_subset

        self._yaw_evaluation_grid = evaluation_grid
        return evaluation_grid

    def _process_evaluation_grid(self):
        # Evaluate the farm AEPs for the grid of possible yaw angles
        evaluation_grid = self._yaw_evaluation_grid
        farm_powers = self._calc_powers_with_memory(evaluation_grid)
        return farm_powers

    def _optimize_sr(self, print_progress=True):
        """
        Find the yaw angles that maximize the power production for every wind direction,
        wind speed and turbulence intensity using the SR optimization algorithm.
        """
        self.print_progress = print_progress

        # For each pass, from front to back
        ii = 0
        for Nii in range(len(self.Ny_passes)):
            # Disturb yaw angles for one turbine at a time, from front to back
            for turbine_depth in range(self.nturbs):
                p = 100.0 * ii / (len(self.Ny_passes) * self.nturbs)
                ii += 1
                if self.print_progress:
                    print(
                        f"[Serial Refine] Processing pass={Nii}, "
                        f"turbine_depth={turbine_depth} ({p:.1f}%)"
                    )

                # Create grid to evaluate yaw angles for one turbine == turbine_depth
                evaluation_grid = self._generate_evaluation_grid(
                    pass_depth=Nii,
                    turbine_depth=turbine_depth
                )

                # Evaluate grid of yaw angles, get farm powers and find optimal solutions
                farm_powers = self._process_evaluation_grid()

                # If farm powers contains any nans, then issue a warning
                if np.any(np.isnan(farm_powers)):
                    err_msg = (
                        "NaNs found in farm powers during SerialRefine "
                        "optimization routine. Proceeding to maximize over yaw "
                        "settings that produce valid powers."
                    )
                    self.logger.warning(err_msg, stack_info=True)

                # Find optimal solutions in new evaluation grid
                args_opt = np.expand_dims(np.nanargmax(farm_powers, axis=0), axis=0)
                farm_powers_opt_new = np.squeeze(
                    np.take_along_axis(farm_powers, args_opt, axis=0),
                    axis=0,
                )
                yaw_angles_opt_new = np.squeeze(
                    np.take_along_axis(
                        evaluation_grid,
                        np.expand_dims(args_opt, axis=2),
                        axis=0
                    ),
                    axis=0
                )

                farm_powers_opt_prev = self._farm_power_opt_subset
                yaw_angles_opt_prev = self._yaw_angles_opt_subset

                # Now update optimal farm powers if better than previous
                ids_better = (farm_powers_opt_new > farm_powers_opt_prev)
                farm_power_opt = farm_powers_opt_prev
                farm_power_opt[ids_better] = farm_powers_opt_new[ids_better]

                # Now update optimal yaw angles if better than previous
                turbs_sorted = self.turbines_ordered_array_subset
                turbids = turbs_sorted[np.where(ids_better)[0], turbine_depth]
                ids = (*np.where(ids_better), turbids)
                yaw_angles_opt = yaw_angles_opt_prev
                yaw_angles_opt[ids] = yaw_angles_opt_new[ids]

                # Update bounds for next iteration to close proximity of optimal solution
                dx = (
                    evaluation_grid[1, :, :] -
                    evaluation_grid[0, :, :]
                )[ids]
                self._yaw_lbs[ids] = np.clip(
                    yaw_angles_opt[ids] - 0.50 * dx,
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )
                self._yaw_ubs[ids] = np.clip(
                    yaw_angles_opt[ids] + 0.50 * dx,
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )

                # Save results to self
                self._farm_power_opt_subset = farm_power_opt
                self._yaw_angles_opt_subset = yaw_angles_opt

        # Finalize optimization, i.e., retrieve full solutions
        df_opt = self._finalize()
        return df_opt

    def _generate_evaluation_grid_gsr(self, pass_depth, turbine_depth):
        """Recompute every turbine's interval, independent of power improvement.

        pass_depth = p - 1. Radius shrinks from the original half-width, not
        from a clipped interval or from the count of successful improvements.
        Do not round the radius: round candidate endpoints inward below.
        """
        rows = np.arange(self._n_findex_subset)
        turbines = self.turbines_ordered_array_subset[:, turbine_depth]
        ids = (rows, turbines)
        lb = self._minimum_yaw_angle_subset[ids]
        ub = self._maximum_yaw_angle_subset[ids]
        current = self._yaw_angles_opt_subset[ids]
        if pass_depth == 0 or self.gsr_alpha == 1.0:
            lower, upper = lb.copy(), ub.copy()
        else:
            initial_half_width = (ub - lb) / 2.0
            # p=2 -> half-width/alpha; p=3 -> half-width/alpha**2.
            # Negative exponent avoids overflow for many passes / large alpha.
            radius = initial_half_width * self.gsr_alpha ** (-pass_depth)
            lower = np.maximum(lb, current - radius)
            upper = np.minimum(ub, current + radius)
        self._yaw_lbs[ids], self._yaw_ubs[ids] = lower, upper
        candidates = []
        for lo, hi, incumbent in zip(lower, upper, current):
            values = np.arange(np.ceil(lo), np.floor(hi) + 1.0)
            if values.size == 0:
                values = np.array([lo, hi])
            candidates.append(np.unique(np.append(values, incumbent)))
        grid = np.tile(self._yaw_angles_opt_subset,
                       (max(map(len, candidates)), 1, 1))
        for row, turbine, values in zip(rows, turbines, candidates):
            grid[:len(values), row, turbine] = values
        self._yaw_evaluation_grid = grid
        return grid

    def _optimize_gsr(self, print_progress=True):
        self.print_progress = print_progress
        rows = np.arange(self._n_findex_subset)
        for depth in range(self.gsr_n_passes):
            for turbine_depth in range(self.nturbs):
                if print_progress:
                    print(f"[GSR pass-based] pass={depth + 1}/{self.gsr_n_passes}, "
                          f"turbine={turbine_depth + 1}/{self.nturbs}")
                grid = self._generate_evaluation_grid_gsr(depth, turbine_depth)
                powers = self._calc_powers_with_memory(grid)
                valid = np.isfinite(powers)
                if not np.all(valid):
                    self.logger.warning("Non-finite GSR candidates ignored; retaining incumbent "
                                        "where no valid improvement is available.")
                scores = np.where(valid, powers, -np.inf)
                best = np.argmax(scores, axis=0)
                proposed = scores[best, rows]
                improved = proposed > self._farm_power_opt_subset
                selected_rows = rows[improved]
                self._farm_power_opt_subset[improved] = proposed[improved]
                self._yaw_angles_opt_subset[improved] = grid[best[improved], selected_rows]
        return self._finalize()

    def optimize(self, print_progress=True):
        """Return the standard FLORIS optimization DataFrame.

        mode='sr': original SR, controlled by Ny_passes (default [5, 4]).
        mode='gsr': unconditional pass refinement; gsr_n_passes=3, gsr_alpha=2.
        Each call performs the configured passes starting from the stored incumbent.
        """
        if self.mode == "sr":
            return self._optimize_sr(print_progress)
        return self._optimize_gsr(print_progress)
