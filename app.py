import matplotlib.pyplot as plt
import numpy as np
from shiny import ui, render, App, reactive
import random
from random import shuffle

current_state = {
    "step": 0,
    "patients": [],
    "treated_patients": [],
    "last_10_steps_removed": [],
    "median_wait_time_treated": [],
    "median_wait_time_untreated": [],
    "mean_wait_time_treated": [],
    "mean_wait_time_untreated": [],
    "percentile_90_treated": [],
    "percentile_10_treated": [],
    "percentile_90_untreated": [],
    "percentile_10_untreated": []
}

class Patient:
    def __init__(self, x1, x2, x3, x4, x5):
        self.wait_time = 0

        # Normalize the weights to ensure they sum to 1
        total = x1 + x2 + x3 + x4 + x5
        weights = [x1/total, x2/total, x3/total, x4/total, x5/total]
        
        # Choose a random interval according to the provided weights
        interval = np.random.choice(range(5), p=weights)
        
        # Generate a random number in the selected interval
        if interval == 0:
            self.urgency = 0.1
        elif interval == 1:
            self.urgency = 0.3
        elif interval == 2:
            self.urgency = 0.5
        elif interval == 3:
            self.urgency = 0.7
        else:
            self.urgency = 0.9

    def increase_urgency(self, func, severity_increase):
        self.urgency = func(self.urgency, self.wait_time, severity_increase)
        self.wait_time += 1

def urgency_increase_function(urgency, wait_time, severity_increase):
    return urgency + severity_increase

def add_new_patients(patients, n, x1, x2, x3, x4, x5):
    for _ in range(n):
        patients.append(Patient(x1, x2, x3, x4, x5))

def remove_treated_patients(patients, n, treated_patients):
    current_state["treated_patients"].extend(patients[:n])
    return patients[n:]

def calculate_median_wait_time(patients):
    return np.median([patient.wait_time for patient in patients])

def calculate_mean_wait_time(patients):
    return np.mean([patient.wait_time for patient in patients])

app_ui = ui.page_fluid(
    ui.h1("Patient Waiting Time Simulator"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_action_button("run", "Run simulation"),
            ui.input_action_button("next_step", "Step through"),
            ui.input_slider("initial_patients", "Initial Patients", 0, 2000, 1000),
            ui.input_slider("new_patients", "New Patients", 0, 100, 50),
            ui.input_slider("removed_patients", "Removed Patients", 0, 100, 50),
            ui.input_slider("time_steps", "Time Steps", 10, 500, 52),
            ui.input_slider("severity_increase", "Severity increase per time step", 0, 0.1, 0.1),
            ui.markdown("""
                        Adjust random severity weighting.\n
                        Weightings are normalised so while the values don't
                        NEED to sum to 1, it's most intuitive to make them sum to 1.
                    """),
            ui.input_numeric("x1", "Severity = 1 weighting", value=0.2),
            ui.input_numeric("x2", "Severity = 2 weighting", value=0.2),
            ui.input_numeric("x3", "Severity = 3 weighting", value=0.2),
            ui.input_numeric("x4", "Severity = 4 weighting", value=0.2),
            ui.input_numeric("x5", "Severity = 5 weighting", value=0.2)
        ),
        ui.panel_main(
            ui.output_plot("plot", width="100%" ,height="1500px")
        )
    )
)

def server(input, output, session):
    def simulate_one_step():
        global current_state

        step = current_state["step"]
        patients = current_state["patients"]
        treated_patients = current_state["treated_patients"]
        last_10_steps_removed = current_state["last_10_steps_removed"]
        median_wait_time_treated = current_state["median_wait_time_treated"]
        median_wait_time_untreated = current_state["median_wait_time_untreated"]
        mean_wait_time_treated = current_state["mean_wait_time_treated"]
        mean_wait_time_untreated = current_state["mean_wait_time_untreated"]
        percentile_90_treated = current_state["percentile_90_treated"]
        percentile_10_treated = current_state["percentile_10_treated"]
        percentile_90_untreated = current_state["percentile_90_untreated"]
        percentile_10_untreated = current_state["percentile_10_untreated"]

        for patient in patients:
            patient.increase_urgency(urgency_increase_function, input.severity_increase())

        shuffle(patients)
        patients.sort(key=lambda p: -p.urgency)

        if step >= input.time_steps() - 10:
                last_10_steps_removed.extend(patients[:input.removed_patients()])

        add_new_patients(patients, input.new_patients(), input.x1(), input.x2(), input.x3(), input.x4(), input.x5())

        patients = remove_treated_patients(patients, input.removed_patients(), treated_patients)

        median_wait_time_treated.append(calculate_median_wait_time(treated_patients))
        median_wait_time_untreated.append(calculate_median_wait_time(patients))
        mean_wait_time_treated.append(calculate_mean_wait_time(treated_patients))
        mean_wait_time_untreated.append(calculate_mean_wait_time(patients))
        percentile_90_treated.append(np.percentile([patient.wait_time for patient in treated_patients], 90) if treated_patients else np.nan)
        percentile_10_treated.append(np.percentile([patient.wait_time for patient in treated_patients], 10) if treated_patients else np.nan)
        percentile_90_untreated.append(np.percentile([patient.wait_time for patient in patients], 90) if patients else np.nan)
        percentile_10_untreated.append(np.percentile([patient.wait_time for patient in patients], 10) if patients else np.nan)

        current_state["step"] += 1
        current_state["patients"] = patients
        current_state["treated_patients"] = treated_patients
        current_state["last_10_steps_removed"] = last_10_steps_removed
        current_state["median_wait_time_treated"] = median_wait_time_treated
        current_state["median_wait_time_untreated"] = median_wait_time_untreated
        current_state["mean_wait_time_treated"] = mean_wait_time_treated
        current_state["mean_wait_time_untreated"] = mean_wait_time_untreated
        current_state["percentile_90_treated"] = percentile_90_treated
        current_state["percentile_10_treated"] = percentile_10_treated
        current_state["percentile_90_untreated"] = percentile_90_untreated
        current_state["percentile_10_untreated"] = percentile_10_untreated

    @reactive.Effect
    @reactive.event(input.next_step)
    def handle_next_step():
        print("Starting single step...")
        simulate_one_step()
        print("Finished single step.")

    @reactive.Effect
    @reactive.event(input.run, ignore_none=False)
    def handle_run_event():
        print("Starting full run...")
        global current_state

        # Reset the state
        current_state["step"] = 0
        current_state["patients"] = [Patient(input.x1(), input.x2(), input.x3(), input.x4(), input.x5()) for _ in range(input.initial_patients())]
        current_state["treated_patients"] = [] 
        current_state["last_10_steps_removed"] = []
        current_state["median_wait_time_treated"] = []
        current_state["median_wait_time_untreated"] = []
        current_state["mean_wait_time_treated"] = []
        current_state["mean_wait_time_untreated"] = []
        current_state["percentile_90_treated"] = []
        current_state["percentile_10_treated"] = []
        current_state["percentile_90_untreated"] = []
        current_state["percentile_10_untreated"] = []

        # Run the simulation for the number of time steps
        for _ in range(input.time_steps()):
            simulate_one_step()

        print("Finished full run.")
        render.plot()

    def plot_graphs():
        step = current_state["step"]
        patients = current_state["patients"]
        treated_patients = current_state["treated_patients"]
        last_10_steps_removed = current_state["last_10_steps_removed"]
        median_wait_time_treated = current_state["median_wait_time_treated"]
        median_wait_time_untreated = current_state["median_wait_time_untreated"]
        mean_wait_time_treated = current_state["mean_wait_time_treated"]
        mean_wait_time_untreated = current_state["mean_wait_time_untreated"]
        percentile_90_treated = current_state["percentile_90_treated"]
        percentile_10_treated = current_state["percentile_10_treated"]
        percentile_90_untreated = current_state["percentile_90_untreated"]
        percentile_10_untreated = current_state["percentile_10_untreated"]

        fig, axs = plt.subplots(3, 2, figsize=(12, 12))

        # Treated Patients: Median wait time plot with shaded percentiles
        axs[0, 0].plot(range(len(median_wait_time_treated)), median_wait_time_treated, color='blue', label='Treated Patients - Median')
        axs[0, 0].fill_between(range(len(median_wait_time_treated)), percentile_10_treated, percentile_90_treated, color='blue', alpha=0.3, label='Treated Patients - 10th to 90th Percentile')
        axs[0, 0].set_xlabel('Time Step')
        axs[0, 0].set_ylabel('Wait Time')
        axs[0, 0].legend()
        axs[0, 0].set_title('Treated Patients: Median/10th/90th Percentiles')
        axs[0, 0].grid(True)

        # Untreated Patients: Median wait time plot with shaded percentiles
        axs[0, 1].plot(range(len(median_wait_time_untreated)), median_wait_time_untreated, color='red', label='Untreated Patients - Median')
        axs[0, 1].fill_between(range(len(median_wait_time_untreated)), percentile_10_untreated, percentile_90_untreated, color='red', alpha=0.3, label='Untreated Patients - 10th to 90th Percentile')
        axs[0, 1].set_xlabel('Time Step')
        axs[0, 1].set_ylabel('Wait Time')
        axs[0, 1].legend()
        axs[0, 1].set_title('Untreated Patients: Median/10th/90th Percentiles')
        axs[0, 1].grid(True)

        # Wait time distribution plot for untreated patients
        all_wait_times = [patient.wait_time for patient in patients]

        # Extract all urgencies
        all_urgencies = [round(patient.urgency, 3) for patient in patients]

        # Convert to a set to remove duplicates and then back to a list
        all_urgencies = list(set(all_urgencies))

        # Sort in descending order and take the top 5
        unique_urgencies = sorted(all_urgencies, reverse=True)[:5]

        # Colors to represent the urgencies
        colors = ['red', 'orange', 'yellow', 'green', 'blue']

        if all_wait_times:
            # Compute the histogram for all wait times
            all_hist_values, bins = np.histogram(all_wait_times, bins=range(max(all_wait_times) + 2))
            
            # Width for each bar
            width = 0.8  # you can adjust this value if needed

            # Plot histogram for all patients first, so it's at the background
            axs[1, 0].bar(bins[:-1], all_hist_values, width=width, color='grey', label='All patients', edgecolor='black')

            # This will keep track of the base for each bar, initialized to zeros
            cumulative_hist = np.zeros_like(all_hist_values)

            # Plot histograms for top urgencies
            for i, urgency in enumerate(unique_urgencies):
                # Get wait times for patients with the specific urgency
                urgent_wait_times = [patient.wait_time for patient in patients if round(patient.urgency, 3) == urgency]
                # Compute the histogram for these wait times
                urgent_hist_values, _ = np.histogram(urgent_wait_times, bins=range(max(all_wait_times) + 2))
                
                # Plot histogram with the base set as the cumulative histogram
                axs[1, 0].bar(bins[:-1], urgent_hist_values, width=width, color=colors[i], label=f'Urgency={urgency:.3f}', edgecolor='black', bottom=cumulative_hist)
                
                # Update the cumulative histogram for the next urgency level
                cumulative_hist += urgent_hist_values

            axs[1, 0].set_xlabel('Wait Time (Time Steps)')
            axs[1, 0].set_ylabel('Frequency')
            axs[1, 0].set_title('Distribution of Wait Times for Untreated')
            axs[1, 0].grid(True)
            axs[1, 0].legend()  # to show the legend

        # Wait time distribution plot for patients removed at last time step
        last_step_removed_wait_times = [patient.wait_time for patient in last_10_steps_removed]
        if last_step_removed_wait_times:
            max_wait_time = max(last_step_removed_wait_times)
            bins = range(max_wait_time + 2) # Adding 1 to ensure that max wait time is included
            counts, _ = np.histogram(last_step_removed_wait_times, bins=bins)
            
            # Calculate the average counts over the last 10 time steps
            avg_counts = counts / 10.0
            
            axs[1, 1].bar(bins[:-1], avg_counts, align='center', edgecolor='black')
            axs[1, 1].set_xlabel('Wait Time (Time Steps)')
            axs[1, 1].set_ylabel('Average Frequency')
            axs[1, 1].set_title('Average Distribution of Wait Times for Treated Patients at Last 10 Time Steps')
            axs[1, 1].grid(True)

        # Mean wait time plot
        axs[2, 0].plot(range(len(mean_wait_time_treated)), mean_wait_time_treated, label='Treated Patients')
        axs[2, 0].plot(range(len(mean_wait_time_untreated)), mean_wait_time_untreated, label='Untreated Patients')
        axs[2, 0].set_xlabel('Time Step')
        axs[2, 0].set_ylabel('Mean Wait Time')
        axs[2, 0].legend()
        axs[2, 0].set_title('Mean Wait Time Over Time')
        axs[2, 0].grid(True)

        return fig

    @output
    @render.plot
    def plot():
        input.run()
        input.next_step()
        print("Plotting")
        print("Finished plot.")
        return plot_graphs()
        

app = App(app_ui, server)
