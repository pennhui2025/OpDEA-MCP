library(shiny)

# Define UI for the app
ui1 <- fluidPage(

  # App title
  titlePanel("Simple Shiny App: Random Numbers"),

  # Sidebar layout with input and output
  sidebarLayout(

    # Sidebar panel for inputs
    sidebarPanel(

      # Input: Numeric input for number of random numbers to generate
      numericInput(inputId = "num",
                   label = "Choose a number of random numbers to generate:",
                   value = 25,
                   min = 1,
                   max = 100)
    ),

    # Main panel for displaying outputs
    mainPanel(

      # Output: Plot
      plotOutput(outputId = "distPlot")
    )
  )
)

# Define server logic for the app
server1 <- function(input, output) {

  # Create the plot
  output$distPlot <- renderPlot({

    # Generate random numbers based on the user's input
    random_numbers <- rnorm(input$num)

    # Plot histogram of the random numbers
    hist(random_numbers,
         col = 'lightblue',
         border = 'white',
         main = "Histogram of Random Numbers",
         xlab = "Random Numbers",
         ylab = "Frequency")
  })
}
