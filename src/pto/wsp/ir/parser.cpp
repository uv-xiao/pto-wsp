// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - IR Parser Implementation
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: MIT

#include "pto/wsp/ir/parser.hpp"
#include <fstream>
#include <sstream>
#include <cctype>

namespace pto::wsp::ir {

// ============================================================
// Keywords Table
// ============================================================

const std::unordered_map<std::string, TokenType>& Lexer::keywords() {
    static const std::unordered_map<std::string, TokenType> kw = {
        {"module", TokenType::KwModule},
        {"version", TokenType::KwVersion},
        {"target", TokenType::KwTarget},
        {"workload", TokenType::KwWorkload},
        {"schedule", TokenType::KwSchedule},
        {"pipeline", TokenType::KwPipeline},
        {"parallel_for", TokenType::KwParallelFor},
        {"for_each", TokenType::KwForEach},
        {"select", TokenType::KwSelect},
        {"cond", TokenType::KwCond},
        {"task", TokenType::KwTask},
        {"combine", TokenType::KwCombine},
        {"sequential", TokenType::KwSequential},
        {"call", TokenType::KwCall},
        {"channel", TokenType::KwChannel},
        {"process", TokenType::KwProcess},
        {"send", TokenType::KwSend},
        {"consume", TokenType::KwConsume},
        {"dispatch", TokenType::KwDispatch},
        {"streams", TokenType::KwStreams},
        {"stream_by", TokenType::KwStreamBy},
        {"timing", TokenType::KwTiming},
        {"spatial_map", TokenType::KwSpatialMap},
        {"layout", TokenType::KwLayout},
        {"dense", TokenType::KwDense},
        {"Dense", TokenType::KwDense},  // Case variant
        {"dense_dyn", TokenType::KwDenseDyn},
        {"ragged", TokenType::KwRagged},
        {"sparse", TokenType::KwSparse},
        {"in", TokenType::KwIn},
        {"for", TokenType::KwFor},
        {"as", TokenType::KwAs},
        {"else", TokenType::KwElse},
        {"yield", TokenType::KwYield},
        {"resources", TokenType::KwResources},
        {"consumes", TokenType::KwConsumes},
        {"produces", TokenType::KwProduces},
        // Schedule policy keywords
        {"round_robin", TokenType::KwRoundRobin},
        {"affinity", TokenType::KwAffinity},
        {"hash", TokenType::KwHash},
        {"work_steal", TokenType::KwWorkSteal},
        {"immediate", TokenType::KwImmediate},
        {"batched", TokenType::KwBatched},
        {"interleaved", TokenType::KwInterleaved},
        {"rate_limit", TokenType::KwRateLimit},
        {"pipeline_depth", TokenType::KwPipelineDepth},
        {"task_window", TokenType::KwTaskWindow},
        {"batch_deps", TokenType::KwBatchDeps},
    };
    return kw;
}

// ============================================================
// Lexer Implementation
// ============================================================

Lexer::Lexer(std::istream& is) : is_(is) {
    // Prime the lexer by reading first character
    advance();
}

int Lexer::advance() {
    int prev = current_;
    current_ = is_.get();
    if (prev == '\n') {
        line_++;
        column_ = 1;
    } else if (prev != -1) {
        column_++;
    }
    return prev;
}

int Lexer::peek() {
    return current_;
}

void Lexer::consumeChar() {
    advance();
}

void Lexer::skipWhitespace() {
    while (current_ != -1) {
        if (current_ == ' ' || current_ == '\t' || current_ == '\r') {
            advance();
        } else if (current_ == '/') {
            // Check for line comment
            int next = is_.peek();
            if (next == '/') {
                skipLineComment();
            } else {
                break;
            }
        } else {
            break;
        }
    }
}

void Lexer::skipLineComment() {
    // Skip // and rest of line
    while (current_ != -1 && current_ != '\n') {
        advance();
    }
}

Token Lexer::next() {
    skipWhitespace();

    int startLine = line_;
    int startCol = column_;

    if (current_ == -1) {
        return Token{TokenType::Eof, "", startLine, startCol};
    }

    // Newline
    if (current_ == '\n') {
        advance();
        return Token{TokenType::Newline, "\n", startLine, startCol};
    }

    // Single-character tokens
    switch (current_) {
        case '@': advance(); return Token{TokenType::At, "@", startLine, startCol};
        case ':': advance(); return Token{TokenType::Colon, ":", startLine, startCol};
        case ',': advance(); return Token{TokenType::Comma, ",", startLine, startCol};
        case '(': advance(); return Token{TokenType::LParen, "(", startLine, startCol};
        case ')': advance(); return Token{TokenType::RParen, ")", startLine, startCol};
        case '{': advance(); return Token{TokenType::LBrace, "{", startLine, startCol};
        case '}': advance(); return Token{TokenType::RBrace, "}", startLine, startCol};
        case '[': advance(); return Token{TokenType::LBracket, "[", startLine, startCol};
        case ']': advance(); return Token{TokenType::RBracket, "]", startLine, startCol};
        case '=': advance(); return Token{TokenType::Equals, "=", startLine, startCol};
        case '|': advance(); return Token{TokenType::Pipe, "|", startLine, startCol};
        case '%': advance(); return Token{TokenType::Percent, "%", startLine, startCol};
        case '!': advance(); return Token{TokenType::Exclaim, "!", startLine, startCol};
    }

    // String literal
    if (current_ == '"') {
        advance();  // consume opening quote
        std::string value;
        while (current_ != -1 && current_ != '"') {
            if (current_ == '\\') {
                advance();
                if (current_ == -1) break;
                switch (current_) {
                    case 'n': value += '\n'; break;
                    case 't': value += '\t'; break;
                    case 'r': value += '\r'; break;
                    case '\\': value += '\\'; break;
                    case '"': value += '"'; break;
                    default: value += static_cast<char>(current_); break;
                }
            } else {
                value += static_cast<char>(current_);
            }
            advance();
        }
        if (current_ == '"') advance();  // consume closing quote
        return Token{TokenType::String, value, startLine, startCol};
    }

    // Number (integer or float)
    if (std::isdigit(current_) || current_ == '-') {
        std::string value;
        if (current_ == '-') {
            value += static_cast<char>(current_);
            advance();
        }
        while (std::isdigit(current_)) {
            value += static_cast<char>(current_);
            advance();
        }
        if (current_ == '.') {
            value += static_cast<char>(current_);
            advance();
            while (std::isdigit(current_)) {
                value += static_cast<char>(current_);
                advance();
            }
        }
        // Scientific notation
        if (current_ == 'e' || current_ == 'E') {
            value += static_cast<char>(current_);
            advance();
            if (current_ == '+' || current_ == '-') {
                value += static_cast<char>(current_);
                advance();
            }
            while (std::isdigit(current_)) {
                value += static_cast<char>(current_);
                advance();
            }
        }
        return Token{TokenType::Number, value, startLine, startCol};
    }

    // Identifier or keyword
    if (std::isalpha(current_) || current_ == '_') {
        std::string value;
        while (std::isalnum(current_) || current_ == '_') {
            value += static_cast<char>(current_);
            advance();
        }
        // Check if it's a keyword (case-insensitive)
        std::string lower = value;
        for (auto& c : lower) c = static_cast<char>(std::tolower(c));
        auto& kw = keywords();
        auto it = kw.find(lower);
        if (it != kw.end()) {
            return Token{it->second, value, startLine, startCol};
        }
        return Token{TokenType::Ident, value, startLine, startCol};
    }

    // Unknown character - skip and try again
    advance();
    return next();
}

// ============================================================
// Parser Implementation
// ============================================================

Parser::Parser(std::istream& is) : lexer_(is), factory_() {
    // Prime the parser
    advance();
}

Token Parser::advance() {
    Token prev = current_;
    current_ = lexer_.next();
    // Skip newlines in most contexts
    while (current_.type == TokenType::Newline) {
        current_ = lexer_.next();
    }
    return prev;
}

bool Parser::check(TokenType type) const {
    return current_.type == type;
}

Token Parser::expect(TokenType type, const std::string& msg) {
    if (!check(type)) {
        throw ParseError(current_.line, current_.column, msg);
    }
    return advance();
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

// ============================================================
// Parsing Methods
// ============================================================

IRPtr<AxisNode> Parser::parseAxis() {
    // axis = "dense" "[" number "]"
    //      | "dense_dyn" "(" ident ")"
    //      | "ragged" "(" ident "," ident ")"
    //      | "sparse" "(" ident "," ident "," ident ")"

    if (match(TokenType::KwDense)) {
        expect(TokenType::LBracket, "Expected '[' after 'dense'");
        Token num = expect(TokenType::Number, "Expected size in dense axis");
        expect(TokenType::RBracket, "Expected ']' after size");
        return factory_.create<DenseAxisNode>(std::stoll(num.value));
    }

    if (match(TokenType::KwDenseDyn)) {
        expect(TokenType::LParen, "Expected '(' after 'dense_dyn'");
        Token var = expect(TokenType::Ident, "Expected size variable in dense_dyn axis");
        expect(TokenType::RParen, "Expected ')' after variable");
        return factory_.create<DenseDynAxisNode>(var.value);
    }

    if (match(TokenType::KwRagged)) {
        expect(TokenType::LParen, "Expected '(' after 'ragged'");
        Token outer = expect(TokenType::Ident, "Expected outer size variable");
        expect(TokenType::Comma, "Expected ',' after outer size");
        Token lengths = expect(TokenType::Ident, "Expected lengths variable");
        expect(TokenType::RParen, "Expected ')' after lengths");
        return factory_.create<RaggedAxisNode>(outer.value, lengths.value);
    }

    if (match(TokenType::KwSparse)) {
        expect(TokenType::LParen, "Expected '(' after 'sparse'");
        Token outer = expect(TokenType::Ident, "Expected outer size variable");
        expect(TokenType::Comma, "Expected ',' after outer size");
        Token indptr = expect(TokenType::Ident, "Expected indptr variable");
        expect(TokenType::Comma, "Expected ',' after indptr");
        Token indices = expect(TokenType::Ident, "Expected indices variable");
        expect(TokenType::RParen, "Expected ')' after indices");
        return factory_.create<SparseAxisNode>(outer.value, indptr.value, indices.value);
    }

    throw ParseError(current_.line, current_.column,
                     "Expected axis type (dense, dense_dyn, ragged, sparse)");
}

// Parse a single statement in workload body
IRPtr<WorkloadNode> Parser::parseStatement() {
    // statement = task | parallel_for | for_each | select | cond | combine | sequential | call

    if (match(TokenType::KwTask)) {
        // task @kernel_name(params) resources(resources)
        expect(TokenType::At, "Expected '@' before kernel name");
        Token kernelName = expect(TokenType::Ident, "Expected kernel name");

        std::vector<std::string> params;
        expect(TokenType::LParen, "Expected '(' after kernel name");
        while (!check(TokenType::RParen) && !check(TokenType::Eof)) {
            Token param = expect(TokenType::Ident, "Expected parameter");
            params.push_back(param.value);
            if (!check(TokenType::RParen)) {
                expect(TokenType::Comma, "Expected ',' between parameters");
            }
        }
        expect(TokenType::RParen, "Expected ')' after parameters");

        std::vector<std::string> resources;
        if (match(TokenType::KwResources)) {
            expect(TokenType::LParen, "Expected '(' after 'resources'");
            while (!check(TokenType::RParen) && !check(TokenType::Eof)) {
                Token res = expect(TokenType::Ident, "Expected resource");
                resources.push_back(res.value);
                if (!check(TokenType::RParen)) {
                    expect(TokenType::Comma, "Expected ',' between resources");
                }
            }
            expect(TokenType::RParen, "Expected ')' after resources");
        }

        return factory_.create<TaskNode>(kernelName.value, std::move(params), std::move(resources));
    }

    if (match(TokenType::KwParallelFor)) {
        // parallel_for var in axis { body }
        Token var = expect(TokenType::Ident, "Expected loop variable");
        expect(TokenType::KwIn, "Expected 'in' after loop variable");
        auto axis = parseAxis();
        expect(TokenType::LBrace, "Expected '{' for parallel_for body");
        auto body = parseWorkloadBody();
        expect(TokenType::RBrace, "Expected '}' after parallel_for body");
        return factory_.create<ParallelForNode>(axis, var.value, body);
    }

    if (match(TokenType::KwForEach)) {
        // for_each var in axis { body }
        Token var = expect(TokenType::Ident, "Expected loop variable");
        expect(TokenType::KwIn, "Expected 'in' after loop variable");
        auto axis = parseAxis();
        expect(TokenType::LBrace, "Expected '{' for for_each body");
        auto body = parseWorkloadBody();
        expect(TokenType::RBrace, "Expected '}' after for_each body");
        return factory_.create<ForEachNode>(axis, var.value, body);
    }

    if (match(TokenType::KwSelect)) {
        // select var in sparse { body }
        Token var = expect(TokenType::Ident, "Expected loop variable");
        expect(TokenType::KwIn, "Expected 'in' after loop variable");
        // Must be sparse axis
        if (!match(TokenType::KwSparse)) {
            throw ParseError(current_.line, current_.column, "Expected sparse axis in select");
        }
        expect(TokenType::LParen, "Expected '(' after 'sparse'");
        Token outer = expect(TokenType::Ident, "Expected outer size variable");
        expect(TokenType::Comma, "Expected ',' after outer size");
        Token indptr = expect(TokenType::Ident, "Expected indptr variable");
        expect(TokenType::Comma, "Expected ',' after indptr");
        Token indices = expect(TokenType::Ident, "Expected indices variable");
        expect(TokenType::RParen, "Expected ')' after indices");
        auto sparse = factory_.create<SparseAxisNode>(outer.value, indptr.value, indices.value);

        expect(TokenType::LBrace, "Expected '{' for select body");
        auto body = parseWorkloadBody();
        expect(TokenType::RBrace, "Expected '}' after select body");
        return factory_.create<SelectNode>(sparse, var.value, body);
    }

    if (match(TokenType::KwCond)) {
        // cond predicate { then } else { else }
        Token pred = expect(TokenType::Ident, "Expected predicate");
        expect(TokenType::LBrace, "Expected '{' for then branch");
        auto thenBranch = parseWorkloadBody();
        expect(TokenType::RBrace, "Expected '}' after then branch");
        expect(TokenType::KwElse, "Expected 'else' after then branch");
        expect(TokenType::LBrace, "Expected '{' for else branch");
        auto elseBranch = parseWorkloadBody();
        expect(TokenType::RBrace, "Expected '}' after else branch");
        return factory_.create<CondNode>(pred.value, nullptr, thenBranch, elseBranch);
    }

    if (match(TokenType::KwCombine)) {
        // combine { workloads }
        expect(TokenType::LBrace, "Expected '{' for combine");
        std::vector<IRPtr<WorkloadNode>> workloads;
        while (!check(TokenType::RBrace) && !check(TokenType::Eof)) {
            workloads.push_back(parseStatement());
            match(TokenType::Comma);  // Optional comma between workloads
        }
        expect(TokenType::RBrace, "Expected '}' after combine");
        return factory_.create<CombineNode>(std::move(workloads));
    }

    if (match(TokenType::KwSequential)) {
        // sequential { workloads }
        expect(TokenType::LBrace, "Expected '{' for sequential");
        std::vector<IRPtr<WorkloadNode>> workloads;
        while (!check(TokenType::RBrace) && !check(TokenType::Eof)) {
            workloads.push_back(parseStatement());
            match(TokenType::Comma);  // Optional comma/semicolon between workloads
        }
        expect(TokenType::RBrace, "Expected '}' after sequential");
        return factory_.create<SequentialNode>(std::move(workloads));
    }

    if (match(TokenType::KwCall)) {
        // call @target(args)
        expect(TokenType::At, "Expected '@' before call target");
        Token target = expect(TokenType::Ident, "Expected call target name");
        std::vector<std::string> args;
        expect(TokenType::LParen, "Expected '(' after call target");
        while (!check(TokenType::RParen) && !check(TokenType::Eof)) {
            Token arg = expect(TokenType::Ident, "Expected argument");
            args.push_back(arg.value);
            if (!check(TokenType::RParen)) {
                expect(TokenType::Comma, "Expected ',' between arguments");
            }
        }
        expect(TokenType::RParen, "Expected ')' after arguments");
        return factory_.create<CallNode>(target.value, std::move(args));
    }

    throw ParseError(current_.line, current_.column,
                     "Expected workload statement (task, parallel_for, for_each, select, cond, combine, sequential, call)");
}

// Parse workload body - sequence of statements
IRPtr<WorkloadNode> Parser::parseWorkloadBody() {
    std::vector<IRPtr<WorkloadNode>> statements;

    while (!check(TokenType::RBrace) && !check(TokenType::Eof)) {
        statements.push_back(parseStatement());
    }

    if (statements.empty()) {
        // Return empty combine for empty body
        return factory_.create<CombineNode>(std::vector<IRPtr<WorkloadNode>>{});
    } else if (statements.size() == 1) {
        return statements[0];
    } else {
        // Multiple statements become a combine
        return factory_.create<CombineNode>(std::move(statements));
    }
}

WorkloadDef Parser::parseWorkloadDef() {
    // workload_def = "@" "workload" ("[" level "]")? ident "(" params ")" "{" body "}"
    expect(TokenType::KwWorkload, "Expected 'workload'");

    // Optional level annotation [npu]
    WorkloadLevel level = WorkloadLevel::CPU;
    if (match(TokenType::LBracket)) {
        Token lvl = expect(TokenType::Ident, "Expected level (cpu or npu)");
        if (lvl.value == "npu") {
            level = WorkloadLevel::NPU;
        } else if (lvl.value != "cpu") {
            throw ParseError(lvl.line, lvl.column, "Unknown level: " + lvl.value);
        }
        expect(TokenType::RBracket, "Expected ']' after level");
    }

    Token name = expect(TokenType::Ident, "Expected workload name");

    // Parameters
    std::vector<std::pair<std::string, IRPtr<AxisNode>>> params;
    expect(TokenType::LParen, "Expected '(' after workload name");
    while (!check(TokenType::RParen) && !check(TokenType::Eof)) {
        expect(TokenType::Percent, "Expected '%' before parameter name");
        Token paramName = expect(TokenType::Ident, "Expected parameter name");
        expect(TokenType::Colon, "Expected ':' after parameter name");
        auto axis = parseAxis();
        params.emplace_back(paramName.value, axis);

        if (!check(TokenType::RParen)) {
            expect(TokenType::Comma, "Expected ',' between parameters");
        }
    }
    expect(TokenType::RParen, "Expected ')' after parameters");

    expect(TokenType::LBrace, "Expected '{' for workload body");
    // Parse the body
    auto body = parseWorkloadBody();
    expect(TokenType::RBrace, "Expected '}' after workload body");

    WorkloadDef def;
    def.name = name.value;
    def.level = level;
    def.params = std::move(params);
    def.body = body;
    return def;
}

IRPtr<WorkloadNode> Parser::parseWorkload() {
    // This is kept for compatibility but not used in current Module structure
    WorkloadDef def = parseWorkloadDef();
    return nullptr;
}

// ============================================================
// Schedule Parsing
// ============================================================

IRPtr<ScheduleNode> Parser::parseScheduleDirective() {
    // dispatch = round_robin(n) | affinity(expr) | hash(expr) | work_steal
    if (match(TokenType::KwDispatch)) {
        expect(TokenType::Equals, "Expected '=' after 'dispatch'");

        if (match(TokenType::KwRoundRobin)) {
            expect(TokenType::LParen, "Expected '(' after 'round_robin'");
            Token num = expect(TokenType::Number, "Expected number in round_robin");
            expect(TokenType::RParen, "Expected ')' after number");
            return factory_.create<DispatchNode>(DispatchPolicy::RoundRobin, std::stoi(num.value), "");
        }

        if (match(TokenType::KwAffinity)) {
            expect(TokenType::LParen, "Expected '(' after 'affinity'");
            Token expr = expect(TokenType::Ident, "Expected expression in affinity");
            expect(TokenType::RParen, "Expected ')' after expression");
            return factory_.create<DispatchNode>(DispatchPolicy::Affinity, 0, expr.value);
        }

        if (match(TokenType::KwHash)) {
            expect(TokenType::LParen, "Expected '(' after 'hash'");
            Token expr = expect(TokenType::Ident, "Expected expression in hash");
            expect(TokenType::RParen, "Expected ')' after expression");
            return factory_.create<DispatchNode>(DispatchPolicy::Hash, 0, expr.value);
        }

        if (match(TokenType::KwWorkSteal)) {
            return factory_.create<DispatchNode>(DispatchPolicy::WorkSteal, 0, "");
        }

        throw ParseError(current_.line, current_.column,
                         "Expected dispatch policy (round_robin, affinity, hash, work_steal)");
    }

    // streams = n, stream_by = expr
    if (match(TokenType::KwStreams)) {
        expect(TokenType::Equals, "Expected '=' after 'streams'");
        Token num = expect(TokenType::Number, "Expected number in streams");
        int numStreams = std::stoi(num.value);

        std::string keyExpr;
        // Check for optional stream_by
        if (match(TokenType::KwStreamBy)) {
            expect(TokenType::Equals, "Expected '=' after 'stream_by'");
            Token expr = expect(TokenType::Ident, "Expected expression in stream_by");
            keyExpr = expr.value;
        }
        return factory_.create<StreamNode>(numStreams, keyExpr);
    }

    // timing = immediate | batched(n) | interleaved(n) | rate_limit(n)
    if (match(TokenType::KwTiming)) {
        expect(TokenType::Equals, "Expected '=' after 'timing'");

        if (match(TokenType::KwImmediate)) {
            return factory_.create<TimingNode>(TimingPolicy::Immediate, 0);
        }

        if (match(TokenType::KwBatched)) {
            expect(TokenType::LParen, "Expected '(' after 'batched'");
            Token num = expect(TokenType::Number, "Expected batch size");
            expect(TokenType::RParen, "Expected ')' after batch size");
            return factory_.create<TimingNode>(TimingPolicy::Batched, std::stoi(num.value));
        }

        if (match(TokenType::KwInterleaved)) {
            expect(TokenType::LParen, "Expected '(' after 'interleaved'");
            Token num = expect(TokenType::Number, "Expected stream count");
            expect(TokenType::RParen, "Expected ')' after stream count");
            return factory_.create<TimingNode>(TimingPolicy::Interleaved, std::stoi(num.value));
        }

        if (match(TokenType::KwRateLimit)) {
            expect(TokenType::LParen, "Expected '(' after 'rate_limit'");
            Token num = expect(TokenType::Number, "Expected rate");
            expect(TokenType::RParen, "Expected ')' after rate");
            return factory_.create<TimingNode>(TimingPolicy::RateLimited, std::stoi(num.value));
        }

        throw ParseError(current_.line, current_.column,
                         "Expected timing policy (immediate, batched, interleaved, rate_limit)");
    }

    // spatial_map = (dim1, dim2, ...)
    if (match(TokenType::KwSpatialMap)) {
        expect(TokenType::Equals, "Expected '=' after 'spatial_map'");
        expect(TokenType::LParen, "Expected '(' for spatial_map dimensions");
        std::vector<int64_t> grid;
        while (!check(TokenType::RParen) && !check(TokenType::Eof)) {
            Token dim = expect(TokenType::Number, "Expected dimension");
            grid.push_back(std::stoll(dim.value));
            if (!check(TokenType::RParen)) {
                expect(TokenType::Comma, "Expected ',' between dimensions");
            }
        }
        expect(TokenType::RParen, "Expected ')' after spatial_map dimensions");
        return factory_.create<SpatialMapNode>(std::move(grid));
    }

    // layout tensor = (S(0), R, ...)
    if (match(TokenType::KwLayout)) {
        Token tensor = expect(TokenType::Ident, "Expected tensor name");
        expect(TokenType::Equals, "Expected '=' after tensor name");
        expect(TokenType::LParen, "Expected '(' for layout dimensions");
        std::vector<LayoutDim> dims;
        while (!check(TokenType::RParen) && !check(TokenType::Eof)) {
            Token dim = expect(TokenType::Ident, "Expected layout dimension (R or S)");
            LayoutDim d;
            if (dim.value == "R") {
                d.kind = LayoutDimKind::Replicate;
                d.mesh_axis = -1;
            } else if (dim.value == "S") {
                expect(TokenType::LParen, "Expected '(' after 'S'");
                Token axis = expect(TokenType::Number, "Expected mesh axis");
                d.kind = LayoutDimKind::Shard;
                d.mesh_axis = std::stoll(axis.value);
                expect(TokenType::RParen, "Expected ')' after mesh axis");
            } else {
                throw ParseError(dim.line, dim.column, "Expected 'R' or 'S' in layout");
            }
            dims.push_back(d);
            if (!check(TokenType::RParen)) {
                expect(TokenType::Comma, "Expected ',' between layout dimensions");
            }
        }
        expect(TokenType::RParen, "Expected ')' after layout dimensions");
        return factory_.create<LayoutNode>(tensor.value, std::move(dims));
    }

    // pipeline_depth = n
    if (match(TokenType::KwPipelineDepth)) {
        expect(TokenType::Equals, "Expected '=' after 'pipeline_depth'");
        Token num = expect(TokenType::Number, "Expected pipeline depth");
        return factory_.create<PipelineDepthNode>(std::stoi(num.value));
    }

    // task_window = n
    if (match(TokenType::KwTaskWindow)) {
        expect(TokenType::Equals, "Expected '=' after 'task_window'");
        Token num = expect(TokenType::Number, "Expected task window size");
        return factory_.create<TaskWindowNode>(std::stoll(num.value));
    }

    // batch_deps = n
    if (match(TokenType::KwBatchDeps)) {
        expect(TokenType::Equals, "Expected '=' after 'batch_deps'");
        Token num = expect(TokenType::Number, "Expected batch deps threshold");
        return factory_.create<BatchDepsNode>(std::stoll(num.value));
    }

    throw ParseError(current_.line, current_.column,
                     "Expected schedule directive (dispatch, streams, timing, spatial_map, layout, pipeline_depth, task_window, batch_deps)");
}

ScheduleDef Parser::parseScheduleDef() {
    // schedule_def = "@" "schedule" ("[" level "]")? ident "for" workload_name "{" directives "}"
    expect(TokenType::KwSchedule, "Expected 'schedule'");

    // Optional level annotation [npu]
    WorkloadLevel level = WorkloadLevel::CPU;
    if (match(TokenType::LBracket)) {
        Token lvl = expect(TokenType::Ident, "Expected level (cpu or npu)");
        if (lvl.value == "npu") {
            level = WorkloadLevel::NPU;
        } else if (lvl.value != "cpu") {
            throw ParseError(lvl.line, lvl.column, "Unknown level: " + lvl.value);
        }
        expect(TokenType::RBracket, "Expected ']' after level");
    }

    Token name = expect(TokenType::Ident, "Expected schedule name");
    expect(TokenType::KwFor, "Expected 'for' after schedule name");
    Token workloadName = expect(TokenType::Ident, "Expected workload name");

    expect(TokenType::LBrace, "Expected '{' for schedule body");

    std::vector<IRPtr<ScheduleNode>> directives;
    while (!check(TokenType::RBrace) && !check(TokenType::Eof)) {
        directives.push_back(parseScheduleDirective());
    }

    expect(TokenType::RBrace, "Expected '}' after schedule body");

    ScheduleDef def;
    def.name = name.value;
    def.workload_name = workloadName.value;
    def.level = level;
    def.directives = std::move(directives);
    return def;
}

// ============================================================
// Header Parsing
// ============================================================

void Parser::parseHeader(Module& m) {
    // header = "module" string
    //        | "version" string
    //        | "target" string ("|" string)*

    while (!check(TokenType::Eof)) {
        if (match(TokenType::KwModule)) {
            Token name = expect(TokenType::String, "Expected module name string");
            m.name = name.value;
        } else if (match(TokenType::KwVersion)) {
            Token ver = expect(TokenType::String, "Expected version string");
            m.version = ver.value;
        } else if (match(TokenType::KwTarget)) {
            // Parse target list: "cpu_sim" | "ascend_npu" | ...
            Token target = expect(TokenType::String, "Expected target string");
            m.targets.push_back(target.value);
            while (match(TokenType::Pipe)) {
                Token next = expect(TokenType::String, "Expected target string after '|'");
                m.targets.push_back(next.value);
            }
        } else {
            break;  // Not a header directive
        }
    }
}

Module Parser::parse() {
    Module m;
    m.version = "9.0";

    // Parse header directives
    parseHeader(m);

    // Parse workloads and schedules
    while (!check(TokenType::Eof)) {
        if (match(TokenType::At)) {
            // @workload or @schedule
            if (check(TokenType::KwWorkload)) {
                m.workloads.push_back(parseWorkloadDef());
            } else if (check(TokenType::KwSchedule)) {
                m.schedules.push_back(parseScheduleDef());
            } else {
                advance();
            }
        } else {
            // Skip unknown tokens
            advance();
        }
    }

    return m;
}

// ============================================================
// Convenience Functions
// ============================================================

Module Parser::parseString(const std::string& source) {
    std::istringstream iss(source);
    Parser parser(iss);
    return parser.parse();
}

Module Parser::parseFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    Parser parser(file);
    return parser.parse();
}

}  // namespace pto::wsp::ir
