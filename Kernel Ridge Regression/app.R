library(shiny)
library(knitr)

# Import the smoothing and regression algorithms
source("kernel_smoothing_regression.R")
# Import the code to create noisy samples from Sin curve
source("noisy_sampling.R")

# Defining custom pastell colours for plotting
col1 <- rgb(27,158,119,max=255)
col2 <- rgb(217,95,2,max=255)
col3 <- rgb(117,112,179,max=255)
colors_3_plot <- c(col1,col2,col3)
names_3_plot  <- c("Samples","Target function", "Kernel smoothing")

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Kernel Smoothing and Kernel Ridge Regression"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        position="right",
        sidebarPanel(
            sliderInput("period_sinoid",
                        "Periodicity of sinoid",
                        min = 0.5,
                        max = 10,
                        value = pi),
            sliderInput("number_points",
                        "Number of noisy samples",
                        min = 10,
                        max = 500,
                        value = 200),
            radioButtons("algorithm", "Algorithm",
                         list("Kernel smoothing", "Kernel Ridge Regression"),
                         inline=TRUE),
            conditionalPanel(
              condition = "input.algorithm == 'Kernel Ridge Regression'",
              sliderInput("lambda",
                          "Regularisation parameter",
                          min = 0.05,
                          max = 10,
                          value = 1.)
            ),
            selectInput("method", "Kernel function",
                        list("Gaussian","Exp-Sine-Squared","RationalQuadratic"),
                        selected = "Exp-Sine-Squared"),
            conditionalPanel(
              condition = "input.method == 'Gaussian'",
              sliderInput("length_gaussian",
                          "Length scale of kernel",
                          min = 0.1,
                          max = 1,
                          value = 0.2)
            ),
            conditionalPanel(
              condition = "input.method == 'Exp-Sine-Squared'",
              sliderInput("length_expsin",
                          "Length scale of kernel",
                          min = 0.1,
                          max = 1,
                          value = 0.2),
              sliderInput("period_expsin",
                          "Periodicity of kernel",
                          min = 0.5,
                          max = 6,
                          value = pi),
            ),
            conditionalPanel(
              condition = "input.method == 'RationalQuadratic'",
              sliderInput("length_ratquad",
                          "Length scale of kernel",
                          min = 0.05,
                          max = 1,
                          value = 0.1),
              sliderInput("alpha_ratquad",
                          "Scale mixture parameter of kernel",
                          min = 0.5,
                          max = 2,
                          value = 1.2),
            )
        ),
  
        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("plot"),
           h3("Description"),
           HTML("<p>This app implements <a href='https://en.wikipedia.org/wiki/Kernel_smoother'><strong>kernel smoothing</strong></a> and <a href='https://scikit-learn.org/stable/modules/kernel_ridge.html'><strong>kernel ridge regression</strong></a> from scratch and demonstrates its use on simulated noisy periodic data. The app's functions are the following:"),
           HTML("<ul><li>Using the first slider, pick a periodicity for a sinus curve, plotted in orange.</li>
                <li>The app then generates a number of noisy samples around this curve, plotted in green. You can change the number of samples using the second slider.</li>
                <li>You can then select between kernel smoothing and kernel ridge regression and determine which kernel to use in the algorithm. The kernel parameters can be adjusted by using the sliders below the dropdown menu.</li>
                <li>We apply the selected algorithm to the noisy data and plot the resulting curve in purple.</li></ul>"),
           HTML("<p>You can pick between the following kernels - see for instance <a href='https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes'>here</a> for a description:"),
           HTML("<ul><li>Gaussian (or Radial Basis Function)</li>
                <li>Exponential Sine Squared</li>
                <li>Rational Quadratic</li></ul>"),
           HTML("<p>The code for the app and the implementation in <code>R</code> of the smoothing and ridge regression algorithms can be found on <a href='https://github.com/maxweissenbacher/max-learning-library/tree/main/Kernel%20Ridge%20Regression%20and%20Smoothing'>GitHub</a>.")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    # Specifying boundaries for sampling
    lower_sample <- 0.
    upper_sample <- 2*pi
    lower_plot <- lower_sample
    upper_plot <- 11
    
    # Generating the true (base) function
    x <- seq(lower_plot,upper_plot,0.05)
    y <- reactive(sinoid(x,input$period_sinoid))
    
    # Generating noisy samples
    samples_fromuser <- reactive(sample_sinoid(n=input$number_points,
                                      input$period_sinoid,
                                      noise = 0.15,
                                      lower_sample,
                                      upper_sample))
    
    # Outputting plot
    output$plot <- renderPlot({
      # Reading samples
      samples <- samples_fromuser()
      
      # Selecting kernel
      if (input$method == "Gaussian") {
        krnl <- gaussian_kernel(sd=input$length_gaussian)
      }
      if (input$method == "Exp-Sine-Squared") {
        krnl <- ExpSineSquared_kernel(input$period_expsin,
                                      input$length_expsin)
      }
      if (input$method == "RationalQuadratic") {
        krnl <- RationalQuadratic_kernel(input$alpha_ratquad,
                                         input$length_ratquad)
      }
      
      # Choosing and applying the correct algorithm
      if (input$algorithm == "Kernel smoothing") {
        smoothed_sinoid <- smoother(samples$x,
                                    samples$y,
                                    krnl)
      }
      if (input$algorithm == "Kernel Ridge Regression") {
        smoothed_sinoid <- kernelRidgeRegression(samples$x,
                                                 samples$y,
                                                 krnl,
                                                 lambda = input$lambda)
      }
      
      # Plotting
      df <- data.frame('x_seq' = x,
                       'true_y' = y(),
                       'smoothed_y' = smoothed_sinoid(x))
      plot <- ggplot(NULL) +
        # Plot the noisy samples
        geom_point(data=samples,aes(x,y,color='Samples'),size=1.6,shape=18) +
        # Plot the true function
        geom_line(data=df, aes(x_seq,true_y,color='Target function'),size=1.0) +
        # Plot the smoothed function
        geom_line(data=df, aes(x_seq,smoothed_y,color='Kernel smoothing'),size=1.0) +
        # Labels and legend
        labs(x = 'x', y = 'f(x)', title = NULL) +
        guides(fill = "none") +
        #theme(aspect.ratio = 1/2) +
        ylim(-1.3,1.3) + xlim(lower_plot,upper_plot) +
        theme(legend.position = "bottom") +
        theme(legend.title=element_blank()) +
        scale_color_manual(values = setNames(colors_3_plot, names_3_plot))

      plot
    })
}



# Run the application 
shinyApp(ui = ui, server = server)
